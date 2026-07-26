#!/usr/bin/env node

/**
 * 批量迷宫生成器
 * 支持所有源码功能：方形、三角形、六边形、圆形迷宫
 * 支持所有算法：递归回溯、Kruskal、Prim等
 */

import { buildMaze } from './js/lib/main.js';
import { algorithms } from './js/lib/algorithms.js';
import {
    SHAPE_SQUARE, SHAPE_TRIANGLE, SHAPE_HEXAGON, SHAPE_CIRCLE,
    ALGORITHM_RECURSIVE_BACKTRACK, ALGORITHM_KRUSKAL, ALGORITHM_SIMPLIFIED_PRIMS,
    ALGORITHM_TRUE_PRIMS, ALGORITHM_WILSON, ALGORITHM_ALDOUS_BRODER, ALGORITHM_HUNT_AND_KILL,
    ALGORITHM_BINARY_TREE, ALGORITHM_SIDEWINDER, ALGORITHM_ELLERS,
    EXITS_NONE, EXITS_HORIZONTAL, EXITS_VERTICAL, EXITS_HARDEST,
    METADATA_START_CELL, METADATA_END_CELL, METADATA_RAW_COORDS, METADATA_PATH,
    DIRECTION_NORTH, DIRECTION_SOUTH, DIRECTION_EAST, DIRECTION_WEST,
    DIRECTION_NORTH_WEST, DIRECTION_NORTH_EAST, DIRECTION_SOUTH_WEST, DIRECTION_SOUTH_EAST,
    DIRECTION_CLOCKWISE, DIRECTION_ANTICLOCKWISE, DIRECTION_INWARDS, DIRECTION_OUTWARDS
} from './js/lib/constants.js';
import { generateShapeAwareMask, generateCellSegmentationMap } from './js/lib/maskGenerator.js';
import fs from 'fs';
import path from 'path';
import { JSDOM } from 'jsdom';
import sharp from 'sharp';
import { pathToFileURL } from 'url';

class BatchMazeGenerator {
    constructor() {
        this.outputDir = './generated_mazes';
        this.solutionDir = './generated_solutions';
        this.noMarkersDir = './generated_mazes_no_markers';  // 无标记版本目录
        this.metadataDir = './generated_metadata';  // 元数据目录
        this.cellSize = 102.4;  // 固定的格子大小（像素）
        this.wallWidth = 10.24;  // 固定的墙壁宽度（像素）
        this.ensureOutputDirs();
    }

    ensureOutputDirs() {
        if (!fs.existsSync(this.outputDir)) {
            fs.mkdirSync(this.outputDir, { recursive: true });
        }
        if (!fs.existsSync(this.solutionDir)) {
            fs.mkdirSync(this.solutionDir, { recursive: true });
        }
        if (!fs.existsSync(this.noMarkersDir)) {
            fs.mkdirSync(this.noMarkersDir, { recursive: true });
        }
        if (!fs.existsSync(this.metadataDir)) {
            fs.mkdirSync(this.metadataDir, { recursive: true });
        }
    }

    /**
     * 根据迷宫配置计算图片尺寸（保持格子大小固定）
     */
    calculateImageSize(shape, width, height, layers) {
        let imageWidth, imageHeight;

        // 获取墙壁宽度用于填充边缘
        const padding = this.wallWidth; 

        if (shape === SHAPE_SQUARE) {
            // 方形迷宫：增加 padding 以容纳右侧和下侧的墙壁
            imageWidth = width * this.cellSize + padding;
            imageHeight = height * this.cellSize + padding;
        } else if (shape === SHAPE_TRIANGLE) {
            // 三角形迷宫：使用与渲染逻辑一致的公式
            const verticalAltitude = Math.sin(Math.PI / 3);
            // 逻辑宽度：0.5 + width/2，逻辑高度：height * verticalAltitude
            imageWidth = (0.5 + width / 2) * this.cellSize + padding;
            imageHeight = height * verticalAltitude * this.cellSize + padding;
        } else if (shape === SHAPE_HEXAGON) {
            // 六边形迷宫：使用与渲染逻辑一致的公式
            const xOffset = Math.sin(Math.PI / 3);
            const yOffset1 = Math.cos(Math.PI / 3);
            const yOffset2 = 2 - yOffset1;
            // 逻辑宽度：width * 2 * xOffset + Math.min(1, height - 1) * xOffset
            // 逻辑高度：height * yOffset2 + yOffset1
            imageWidth = (width * 2 * xOffset + Math.min(1, height - 1) * xOffset) * this.cellSize + padding;
            imageHeight = (height * yOffset2 + yOffset1) * this.cellSize + padding;
        } else if (shape === SHAPE_CIRCLE) {
            // 圆形迷宫：通常居中绘制，增加一点 buffer 防止切边
            const diameter = layers * 2 * this.cellSize;
            imageWidth = diameter + padding * 2;
            imageHeight = diameter + padding * 2;
        } else {
            // 默认
            imageWidth = width * this.cellSize + padding;
            imageHeight = height * this.cellSize + padding;
        }

        // 确保尺寸至少为某个最小值，并且是偶数（便于处理）
        imageWidth = Math.max(200, Math.ceil(imageWidth));
        imageHeight = Math.max(200, Math.ceil(imageHeight));


        // 确保是偶数
        if (imageWidth % 2 !== 0) imageWidth++;
        if (imageHeight % 2 !== 0) imageHeight++;

        return { width: imageWidth, height: imageHeight };
    }

    /**
     * 将SVG转换为PNG
     */
    async convertSvgToPng(svgContent, outputPath, width = 512, height = 512) {
        try {
            // 不使用 resize，让 SVG 按照其定义的尺寸渲染
            // 这样可以保持固定的格子大小和墙壁宽度
            const pngBuffer = await sharp(Buffer.from(svgContent))
                .png()
                .toBuffer();

            fs.writeFileSync(outputPath, pngBuffer);
            return true;
        } catch (error) {
            console.error(`PNG转换失败: ${error.message}`);
            return false;
        }
    }

    /**
     * 生成基于格子的二进制掩码
     * @param {Object} maze - 迷宫对象
     * @param {number} imageWidth - 输出图像宽度
     * @param {number} imageHeight - 输出图像高度
     * @returns {Uint8Array} 二进制掩码数组 (1=解空间, 0=其他)
     */
    generateCellBasedMask(maze, imageWidth = 512, imageHeight = 512) {
        try {
            // 获取解空间格子（路径经过的格子 + 起点终点格子）
            const solutionCells = new Set();

            // 添加路径经过的所有格子
            const path = maze.metadata?.[METADATA_PATH];
            if (path && path.length > 0) {
                path.forEach(coords => {
                    solutionCells.add(coords.join(','));
                });
            }

            // 添加起点和终点格子
            maze.forEachCell(cell => {
                if (cell.metadata[METADATA_START_CELL] || cell.metadata[METADATA_END_CELL]) {
                    solutionCells.add(cell.coords.join(','));
                }
            });

            // 使用新的形状感知掩码生成器
            return generateShapeAwareMask(maze, solutionCells, imageWidth, imageHeight);
        } catch (error) {
            console.error(`生成格子级掩码失败: ${error.message}`);
            return null;
        }
    }

    // /**
    //  * 在掩码中填充格子对应的像素区域
    //  */
    // fillCellRegionInMask(maskData, imageWidth, imageHeight, centerX, centerY, maze) {
    //     const shape = maze.metadata.cellShape;

    //     switch (shape) {
    //         case SHAPE_SQUARE:
    //             this.fillSquareCellMask(maskData, imageWidth, imageHeight, centerX, centerY, maze);
    //             break;
    //         case SHAPE_TRIANGLE:
    //             this.fillTriangleCellMask(maskData, imageWidth, imageHeight, centerX, centerY, maze);
    //             break;
    //         case SHAPE_HEXAGON:
    //             this.fillHexagonCellMask(maskData, imageWidth, imageHeight, centerX, centerY, maze);
    //             break;
    //         case SHAPE_CIRCLE:
    //             this.fillCircleCellMask(maskData, imageWidth, imageHeight, centerX, centerY, maze);
    //             break;
    //         default:
    //             // 默认使用方形逻辑
    //             this.fillSquareCellMask(maskData, imageWidth, imageHeight, centerX, centerY, maze);
    //     }
    // }

    // /**
    //  * 填充方形格子掩码
    //  */
    // fillSquareCellMask(maskData, imageWidth, imageHeight, centerX, centerY, maze) {
    //     const gridWidth = maze.metadata.width;
    //     const gridHeight = maze.metadata.height;

    //     const cellPixelWidth = Math.max(1, Math.floor(imageWidth / gridWidth));
    //     const cellPixelHeight = Math.max(1, Math.floor(imageHeight / gridHeight));

    //     const halfWidth = Math.floor(cellPixelWidth / 2);
    //     const halfHeight = Math.floor(cellPixelHeight / 2);

    //     const startX = Math.max(0, Math.floor(centerX) - halfWidth);
    //     const endX = Math.min(imageWidth - 1, Math.floor(centerX) + halfWidth);
    //     const startY = Math.max(0, Math.floor(centerY) - halfHeight);
    //     const endY = Math.min(imageHeight - 1, Math.floor(centerY) + halfHeight);

    //     for (let y = startY; y <= endY; y++) {
    //         for (let x = startX; x <= endX; x++) {
    //             const index = y * imageWidth + x;
    //             maskData[index] = 1;
    //         }
    //     }
    // }

    // /**
    //  * 填充三角形格子掩码
    //  */
    // fillTriangleCellMask(maskData, imageWidth, imageHeight, centerX, centerY, maze) {
    //     const gridWidth = maze.metadata.width;
    //     const gridHeight = maze.metadata.height;
    //     const verticalAltitude = Math.sin(Math.PI/3);

    //     // 计算三角形的实际尺寸
    //     const cellPixelWidth = imageWidth / (gridWidth/2);
    //     const cellPixelHeight = imageHeight / (gridHeight * verticalAltitude);

    //     // 三角形的边长（基于像素尺寸）
    //     const triangleSize = Math.min(cellPixelWidth, cellPixelHeight) * 0.85;
    //     const triangleHeight = triangleSize * verticalAltitude;

    //     // 为了确定三角形方向，我们需要找到这个格子在网格中的坐标
    //     // 从迷宫数据中找到该格子的坐标
    //     let cellCoords = null;
    //     maze.forEachCell(cell => {
    //         if (cell.metadata[METADATA_RAW_COORDS] &&
    //             Math.abs(cell.metadata[METADATA_RAW_COORDS][0] - centerX) < 1 &&
    //             Math.abs(cell.metadata[METADATA_RAW_COORDS][1] - centerY) < 1) {
    //             cellCoords = cell.coords;
    //         }
    //     });

    //     if (!cellCoords) {
    //         // 如果找不到坐标，使用估算的方向
    //         const estimatedX = Math.round(centerX / cellPixelWidth * 2);
    //         const estimatedY = Math.round(centerY / cellPixelHeight / verticalAltitude);
    //         cellCoords = [estimatedX, estimatedY];
    //     }

    //     // 使用和原始代码相同的逻辑判断三角形方向
    //     const [x, y] = cellCoords;
    //     const hasBaseOnSouthSide = (x + y) % 2 === 1;

    //     // 生成等边三角形的三个顶点
    //     let p1x, p1y, p2x, p2y, p3x, p3y;

    //     if (hasBaseOnSouthSide) {
    //         // 底边朝南，顶点朝上的三角形
    //         p1x = centerX;                          // 顶点（上）
    //         p1y = centerY - triangleHeight * 2/3;
    //         p2x = centerX - triangleSize/2;        // 左下角
    //         p2y = centerY + triangleHeight * 1/3;
    //         p3x = centerX + triangleSize/2;        // 右下角
    //         p3y = centerY + triangleHeight * 1/3;
    //     } else {
    //         // 底边朝北，顶点朝下的三角形
    //         p1x = centerX - triangleSize/2;        // 左上角
    //         p1y = centerY - triangleHeight * 1/3;
    //         p2x = centerX + triangleSize/2;        // 右上角
    //         p2y = centerY - triangleHeight * 1/3;
    //         p3x = centerX;                          // 顶点（下）
    //         p3y = centerY + triangleHeight * 2/3;
    //     }

    //     // 计算三角形的边界框
    //     const minX = Math.max(0, Math.floor(Math.min(p1x, p2x, p3x)));
    //     const maxX = Math.min(imageWidth - 1, Math.ceil(Math.max(p1x, p2x, p3x)));
    //     const minY = Math.max(0, Math.floor(Math.min(p1y, p2y, p3y)));
    //     const maxY = Math.min(imageHeight - 1, Math.ceil(Math.max(p1y, p2y, p3y)));

    //     // 使用点在三角形内的判断算法
    //     for (let y = minY; y <= maxY; y++) {
    //         for (let x = minX; x <= maxX; x++) {
    //             if (this.isPointInTriangle(x, y, p1x, p1y, p2x, p2y, p3x, p3y)) {
    //                 const index = y * imageWidth + x;
    //                 maskData[index] = 1;
    //             }
    //         }
    //     }
    // }

    // /**
    //  * 填充六边形格子掩码
    //  */
    // fillHexagonCellMask(maskData, imageWidth, imageHeight, centerX, centerY, maze) {
    //     const gridWidth = maze.metadata.width;
    //     const gridHeight = maze.metadata.height;

    //     const xOffset = Math.sin(Math.PI / 3);
    //     const yOffset2 = 2 - Math.cos(Math.PI / 3);

    //     const cellPixelWidth = imageWidth / (gridWidth * 2 * xOffset + xOffset);
    //     const cellPixelHeight = imageHeight / (gridHeight * yOffset2);

    //     // 六边形的外接圆半径
    //     const radius = Math.min(cellPixelWidth, cellPixelHeight) * 0.45;

    //     // 生成正六边形的六个顶点
    //     const vertices = [];
    //     for (let i = 0; i < 6; i++) {
    //         const angle = (Math.PI / 3) * i; // 60度间隔
    //         const x = centerX + radius * Math.cos(angle);
    //         const y = centerY + radius * Math.sin(angle);
    //         vertices.push([x, y]);
    //     }

    //     // 计算六边形的边界框
    //     const xs = vertices.map(v => v[0]);
    //     const ys = vertices.map(v => v[1]);
    //     const minX = Math.max(0, Math.floor(Math.min(...xs)));
    //     const maxX = Math.min(imageWidth - 1, Math.ceil(Math.max(...xs)));
    //     const minY = Math.max(0, Math.floor(Math.min(...ys)));
    //     const maxY = Math.min(imageHeight - 1, Math.ceil(Math.max(...ys)));

    //     // 使用点在多边形内的判断算法
    //     for (let y = minY; y <= maxY; y++) {
    //         for (let x = minX; x <= maxX; x++) {
    //             if (this.isPointInPolygon(x, y, vertices)) {
    //                 const index = y * imageWidth + x;
    //                 maskData[index] = 1;
    //             }
    //         }
    //     }
    // }

    // /**
    //  * 填充圆形（环形）格子掩码
    //  */
    // fillCircleCellMask(maskData, imageWidth, imageHeight, centerX, centerY, maze) {
    //     const layers = maze.metadata.layers;

    //     // 圆形迷宫中心
    //     const imageCenterX = imageWidth / 2;
    //     const imageCenterY = imageHeight / 2;

    //     // 每层的半径范围
    //     const maxRadius = Math.min(imageWidth, imageHeight) / 2;
    //     const layerThickness = maxRadius / layers;

    //     // 从格子中心坐标反推出它在哪一层和哪个扇区
    //     const distanceFromCenter = Math.sqrt(
    //         (centerX - imageCenterX) * (centerX - imageCenterX) +
    //         (centerY - imageCenterY) * (centerY - imageCenterY)
    //     );

    //     // 估算这个格子所在的层
    //     const estimatedLayer = Math.floor(distanceFromCenter / layerThickness);

    //     if (estimatedLayer === 0) {
    //         // 中心点，填充一个小圆
    //         const centerRadius = Math.max(4, Math.floor(layerThickness / 4));
    //         const startX = Math.max(0, Math.floor(centerX) - centerRadius);
    //         const endX = Math.min(imageWidth - 1, Math.floor(centerX) + centerRadius);
    //         const startY = Math.max(0, Math.floor(centerY) - centerRadius);
    //         const endY = Math.min(imageHeight - 1, Math.floor(centerY) + centerRadius);

    //         for (let y = startY; y <= endY; y++) {
    //             for (let x = startX; x <= endX; x++) {
    //                 const dx = x - centerX;
    //                 const dy = y - centerY;
    //                 if (dx * dx + dy * dy <= centerRadius * centerRadius) {
    //                     const index = y * imageWidth + x;
    //                     maskData[index] = 1;
    //                 }
    //             }
    //         }
    //     } else {
    //         // 外层格子，生成扇形掩码
    //         const innerRadius = estimatedLayer * layerThickness;
    //         const outerRadius = (estimatedLayer + 1) * layerThickness;

    //         // 计算格子的角度范围
    //         const angleFromCenter = Math.atan2(centerY - imageCenterY, centerX - imageCenterX);

    //         // 估算扇区数量（基于周长）
    //         const circumference = Math.PI * 2 * (innerRadius + outerRadius) / 2;
    //         const estimatedSectors = Math.max(8, Math.round(circumference / layerThickness));
    //         const sectorAngle = Math.PI * 2 / estimatedSectors;

    //         // 格子所在扇区的角度范围
    //         const sectorStartAngle = angleFromCenter - sectorAngle / 2;
    //         const sectorEndAngle = angleFromCenter + sectorAngle / 2;

    //         // 扫描可能的像素区域
    //         const scanRadius = Math.ceil(outerRadius);
    //         const startX = Math.max(0, Math.floor(imageCenterX - scanRadius));
    //         const endX = Math.min(imageWidth - 1, Math.ceil(imageCenterX + scanRadius));
    //         const startY = Math.max(0, Math.floor(imageCenterY - scanRadius));
    //         const endY = Math.min(imageHeight - 1, Math.ceil(imageCenterY + scanRadius));

    //         for (let y = startY; y <= endY; y++) {
    //             for (let x = startX; x <= endX; x++) {
    //                 const dx = x - imageCenterX;
    //                 const dy = y - imageCenterY;
    //                 const pixelRadius = Math.sqrt(dx * dx + dy * dy);

    //                 // 检查是否在正确的环形范围内
    //                 if (pixelRadius >= innerRadius && pixelRadius <= outerRadius) {
    //                     const pixelAngle = Math.atan2(dy, dx);

    //                     // 检查是否在正确的角度扇区内（考虑角度的周期性）
    //                     let angleDiff = pixelAngle - angleFromCenter;
    //                     while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
    //                     while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

    //                     if (Math.abs(angleDiff) <= sectorAngle / 2) {
    //                         const index = y * imageWidth + x;
    //                         maskData[index] = 1;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // /**
    //  * 判断点是否在三角形内（使用重心坐标法）
    //  */
    // isPointInTriangle(px, py, x1, y1, x2, y2, x3, y3) {
    //     const denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    //     if (Math.abs(denom) < 1e-10) return false; // 三角形退化

    //     const a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom;
    //     const b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom;
    //     const c = 1 - a - b;

    //     return a >= 0 && b >= 0 && c >= 0;
    // }

    // /**
    //  * 判断点是否在多边形内（射线法）
    //  */
    // isPointInPolygon(x, y, vertices) {
    //     let inside = false;
    //     for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i++) {
    //         const [xi, yi] = vertices[i];
    //         const [xj, yj] = vertices[j];

    //         if (((yi > y) !== (yj > y)) &&
    //             (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
    //             inside = !inside;
    //         }
    //     }
    //     return inside;
    // }

    /**
     * 将mask数据保存为PNG图像
     */
    async saveMaskAsPNG(maskData, filepath, width = 512, height = 512) {
        try {
            // 将0/1的mask转换为0/255的灰度图
            const grayscaleData = new Uint8Array(maskData.length);
            for (let i = 0; i < maskData.length; i++) {
                grayscaleData[i] = maskData[i] * 255;
            }

            const pngBuffer = await sharp(Buffer.from(grayscaleData), {
                raw: { width, height, channels: 1 }
            }).png().toBuffer();

            fs.writeFileSync(filepath, pngBuffer);
            return true;
        } catch (error) {
            console.error(`Mask PNG保存失败: ${error.message}`);
            return false;
        }
    }

    /**
     * 将格子ID编码为RGB值
     */
    idToRgb(id) {
        const r = (id >> 16) & 0xFF;
        const g = (id >> 8) & 0xFF;
        const b = id & 0xFF;
        return [r, g, b];
    }

    /**
     * 将格子坐标转换为唯一ID
     * 所有有效格子ID从1开始，0保留给背景
     */
    coordsToCellId(coords, maze, shape) {
        if (shape === SHAPE_CIRCLE) {
            // 圆形迷宫：[layer, cellIndex]
            const [layer, cellIndex] = coords;
            // +1确保从1开始
            return layer * 10000 + cellIndex + 1;
        } else {
            // 其他形状：[x, y]
            const [x, y] = coords;
            const width = maze.metadata.width || 100;
            // +1确保从1开始
            return y * width + x + 1;
        }
    }

    /**
     * 将cell map保存为PNG图像（ID编码为RGB，但保存为BGR以匹配OpenCV）
     */
    async saveCellMapAsPNG(cellMap, width, height, filepath) {
        try {
            const rgbBuffer = new Uint8Array(width * height * 3);
            for (let i = 0; i < cellMap.length; i++) {
                const [r, g, b] = this.idToRgb(cellMap[i]);
                // 保存为BGR格式以匹配OpenCV的默认读取顺序
                rgbBuffer[i * 3] = b;     // B通道
                rgbBuffer[i * 3 + 1] = g; // G通道
                rgbBuffer[i * 3 + 2] = r; // R通道
            }

            await sharp(rgbBuffer, {
                raw: { width, height, channels: 3 }
            }).png().toFile(filepath);

            return true;
        } catch (error) {
            console.error(`Cell map PNG保存失败: ${error.message}`);
            return false;
        }
    }

    /**
     * 仅对圆形迷宫：将掩码与无标记迷宫图做按位“相乘”（AND），
     * 以保留黑色墙壁（黑=0，白=1），掩码外区域保持0。
     */
    async multiplyMaskWithNoMarker(maskData, width, height, noMarkerPngPath) {
        try {
            if (!fs.existsSync(noMarkerPngPath)) return maskData;

            // 读取无标记迷宫图像为原始像素
            const img = sharp(noMarkerPngPath);
            const meta = await img.metadata();
            const raw = await img
                .ensureAlpha()
                .raw()
                .toBuffer();

            const channels = meta.channels || 4;
            const pixels = raw; // Uint8Array

            // 将无标记迷宫图转为二值：白(>=200)->1，黑(<200)->0
            const binary = new Uint8Array(width * height);
            const hasAlpha = channels === 4;
            for (let i = 0, p = 0; i < binary.length; i++, p += channels) {
                const r = pixels[p];
                const g = pixels[p + 1];
                const b = pixels[p + 2];
                const a = hasAlpha ? pixels[p + 3] : 255;
                // 简单感知亮度，透明按背景白处理
                const lum = (0.299 * r + 0.587 * g + 0.114 * b) * (a / 255) + 255 * (1 - a / 255);
                binary[i] = lum >= 200 ? 1 : 0;
            }

            // 掩码 AND 无标记图（二值），保留墙线（0）、开放区域（1），掩码外仍为0
            const out = new Uint8Array(maskData.length);
            for (let i = 0; i < maskData.length; i++) {
                out[i] = (maskData[i] & binary[i]) ? 1 : 0;
            }
            return out;
        } catch {
            return maskData;
        }
    }

    /**
     * 计算路径方向序列的信息熵难度
     */
    calculateTurnDifficultyRich(maze, shape, options = {}) {
        const path = maze.metadata?.[METADATA_PATH];
        if (!path || path.length < 2) {
          return { difficulty:0, H_turn:0, H_turn_norm:0, H_rate:0, H_rate_norm:0, turns:[], turnCounts:{}, steps:0 };
        }
      
        // 选择 baseAngle
        const baseAngle =
          (shape === SHAPE_SQUARE) ? 90 :
          (shape === SHAPE_TRIANGLE || shape === SHAPE_HEXAGON) ? 60 :
          90; // circle
      
        // 1) 先把每一步变成 “绝对方向名” 再映射到角度
        const angles = [];
        for (let i = 0; i < path.length - 1; i++) {
          const from = path[i], to = path[i+1];
          let dname = null;
          if (shape === SHAPE_CIRCLE) {
            // 需要传入每层扇区数
            dname = this.circleDirection(from, to, options.segmentsAtLayer);
          } else {
            dname = this.calculateDirection(from, to, shape); // 你已有的函数（建议后续完善六角/三角）
          }
          if (!dname) continue;
          const ang = this.directionAngle(dname, shape);
          if (ang == null) continue;
          angles.push(ang);
        }
        if (angles.length < 2) {
          return { difficulty:0, H_turn:0, H_turn_norm:0, H_rate:0, H_rate_norm:0, turns:[], turnCounts:{}, steps:angles.length };
        }
      
        // 2) 角差 -> 富转向类别
        const turns = [];
        for (let i = 1; i < angles.length; i++) {
          const t = this.angleToRichTurn(angles[i-1], angles[i], baseAngle);
          if (t) turns.push(t);
        }
        if (!turns.length) {
          return { difficulty:0, H_turn:0, H_turn_norm:0, H_rate:0, H_rate_norm:0, turns:[], turnCounts:{}, steps:angles.length };
        }
      
        // 3) H_turn（边缘熵）
        const turnCounts = {};
        for (const t of turns) turnCounts[t] = (turnCounts[t] || 0) + 1;
        const { H: H_turn, m } = this.entropyFromCounts(turnCounts);
        const H_turn_norm = (m > 1) ? H_turn / Math.log2(m) : 0;
      
        // 4) H_rate（熵率：条件熵）
        const trans = {};
        for (let i = 1; i < turns.length; i++) {
          const a = turns[i-1], b = turns[i];
          (trans[a] ??= {}), trans[a][b] = (trans[a][b] || 0) + 1;
        }
        const { H: H_rate } = this.conditionalEntropyFromTransitions(trans);
      
        // 归一化（逐行）
        let H_rate_norm = 0, totalPairs = 0;
        for (const a in trans) for (const b in trans[a]) totalPairs += trans[a][b];
        for (const a in trans) {
          const row = trans[a];
          const rowSum = Object.values(row).reduce((x,y)=>x+y,0);
          const mr = Object.keys(row).length;
          if (!rowSum || mr <= 1) continue;
          let Hr = 0;
          for (const b in row) {
            const p = row[b] / rowSum;
            if (p > 0) Hr -= p * Math.log2(p);
          }
          H_rate_norm += (rowSum / totalPairs) * (Hr / Math.log2(mr));
        }
      
        // 5) 给出可解释的综合分
        const difficulty = 0.6 * H_turn_norm + 0.4 * H_rate_norm;
      
        return {
          difficulty,
          H_turn, H_turn_norm,
          H_rate, H_rate_norm,
          turns,
          turnCounts,
          steps: angles.length,
          baseAngleUsed: baseAngle
        };
      }
      
    
    // ---- 信息熵工具 ----
    entropyFromCounts(counts) {
        const total = Object.values(counts).reduce((a,b)=>a+b, 0);
        if (!total) return { H: 0, m: 0, total: 0 };
        let H = 0, m = 0;
        for (const k in counts) {
        const c = counts[k];
        if (c > 0) {
            const p = c / total;
            H -= p * Math.log2(p);
            m++;
        }
        }
        return { H, m, total };
    }
    
    conditionalEntropyFromTransitions(trans) {
        let totalPairs = 0;
        for (const a in trans) for (const b in trans[a]) totalPairs += trans[a][b];
        if (!totalPairs) return { H: 0, totalPairs: 0 };
    
        let H = 0;
        for (const a in trans) {
        const row = trans[a];
        const rowSum = Object.values(row).reduce((x,y)=>x+y,0);
        if (!rowSum) continue;
        let Hr = 0;
        for (const b in row) {
            const p = row[b] / rowSum;
            if (p > 0) Hr -= p * Math.log2(p);
        }
        H += (rowSum / totalPairs) * Hr;
        }
        return { H, totalPairs };
    }
    
    // ---- 圆形方向（取模）----
    circleDirection(fromCoords, toCoords, segmentsAtLayer) {
        const [fLayer, fIndex] = fromCoords;
        const [tLayer, tIndex] = toCoords;
    
        if (tLayer > fLayer) return 'OUTWARDS';
        if (tLayer < fLayer) return 'INWARDS';
    
        const seg = segmentsAtLayer?.[fLayer];
        if (!seg || seg <= 0) return null;
    
        const diff = ((tIndex - fIndex) % seg + seg) % seg; // 0..seg-1
        if (diff === 0) return null;
        return (diff <= seg/2) ? 'CLOCKWISE' : 'ANTICLOCKWISE';
    }
    
    directionAngle(direction, shape) {
        if (shape === SHAPE_SQUARE) {
          const map = { NORTH:0, EAST:90, SOUTH:180, WEST:270 };
          return map[direction] ?? null;
        }
        if (shape === SHAPE_TRIANGLE || shape === SHAPE_HEXAGON) {
          const map = {
            NORTH:0, NORTH_EAST:60, SOUTH_EAST:120,
            SOUTH:180, SOUTH_WEST:240, NORTH_WEST:300
          };
          return map[direction] ?? null;
        }
        if (shape === SHAPE_CIRCLE) {
          const map = { OUTWARDS:0, CLOCKWISE:90, INWARDS:180, ANTICLOCKWISE:270 };
          return map[direction] ?? null;
        }
        return null;
      }
      
    
    // ---- 角差 -> 富转向类别（LEFT/RIGHT + 角度）----
    angleToRichTurn(prevDeg, currDeg, baseAngle, tolFactor = 1/3) {
        if (prevDeg == null || currDeg == null) return null;
        let d = (currDeg - prevDeg) % 360;
        if (d <= -180) d += 360;
        if (d > 180)  d -= 360;
    
        const ad = Math.abs(d);
        const tol = baseAngle * tolFactor;
    
        if (ad <= tol) return 'STRAIGHT';
        if (Math.abs(ad - 180) <= tol) return 'UTURN_180';
    
        const k = Math.max(1, Math.round(ad / baseAngle));
        const quant = k * baseAngle;
        if (Math.abs(ad - quant) > tol) {
        const dir = d > 0 ? 'RIGHT' : 'LEFT';
        return `OTHER_${dir}_${Math.round(ad)}`;
        }
        const dir = d > 0 ? 'RIGHT' : 'LEFT';
        return `${dir}_${quant}`; // e.g. RIGHT_60, LEFT_120
    }
  

    /**
     * 获取各种形状可用的方向集合
     */
    getAvailableDirections(shape) {
        switch (shape) {
            case SHAPE_SQUARE:
                return ['NORTH', 'EAST', 'SOUTH', 'WEST']; // k=4
            case SHAPE_TRIANGLE:
            case SHAPE_HEXAGON:
                return ['NORTH', 'NORTH_EAST', 'SOUTH_EAST', 'SOUTH', 'SOUTH_WEST', 'NORTH_WEST']; // k=6
            case SHAPE_CIRCLE:
                return ['INWARDS', 'OUTWARDS', 'CLOCKWISE', 'ANTICLOCKWISE']; // k=4
            default:
                return ['NORTH', 'EAST', 'SOUTH', 'WEST'];
        }
    }

    /**
     * 计算两个相邻坐标之间的移动方向
     */
    calculateDirection(fromCoords, toCoords, shape) {
        const [fx, fy] = fromCoords;
        const [tx, ty] = toCoords;
        const dx = tx - fx;
        const dy = ty - fy;

        if (shape === SHAPE_SQUARE) {
            if (dx === 0 && dy === -1) return 'NORTH';
            if (dx === 1 && dy === 0) return 'EAST';
            if (dx === 0 && dy === 1) return 'SOUTH';
            if (dx === -1 && dy === 0) return 'WEST';
        } else if (shape === SHAPE_TRIANGLE || shape === SHAPE_HEXAGON) {
            // 简化的六方向判断
            if (dy < 0) {
                if (dx === 0) return 'NORTH';
                if (dx > 0) return 'NORTH_EAST';
                if (dx < 0) return 'NORTH_WEST';
            } else if (dy > 0) {
                if (dx === 0) return 'SOUTH';
                if (dx > 0) return 'SOUTH_EAST';
                if (dx < 0) return 'SOUTH_WEST';
            } else {
                if (dx > 0) return 'NORTH_EAST';
                if (dx < 0) return 'NORTH_WEST';
            }
        } else if (shape === SHAPE_CIRCLE) {
            // 圆形迷宫：根据层和索引变化判断
            const [fLayer, fIndex] = fromCoords;
            const [tLayer, tIndex] = toCoords;

            if (tLayer > fLayer) return 'OUTWARDS';
            if (tLayer < fLayer) return 'INWARDS';

            if (tLayer === fLayer) {
                const indexDiff = tIndex - fIndex;
                if (indexDiff === 1 || (indexDiff < -1)) return 'CLOCKWISE';
                if (indexDiff === -1 || (indexDiff > 1)) return 'ANTICLOCKWISE';
            }
        }

        return null; // 无法识别的移动
    }

    /**
     * 根据难度值获取难度等级描述
     */
    getDifficultyLevel(difficulty) {
        if (difficulty >= 0.85) return 'EXTREME';
        if (difficulty >= 0.70) return 'HARD';
        if (difficulty >= 0.50) return 'MEDIUM';
        if (difficulty >= 0.30) return 'EASY';
        return 'TRIVIAL';
    }

    /**
     * 生成解决路径
     */
    generateSolution(maze) {
        let startCell = null;
        let endCell = null;

        // 查找入口和出口单元格
        maze.forEachCell(cell => {
            if (cell.metadata[METADATA_START_CELL]) {
                startCell = cell;
            }
            if (cell.metadata[METADATA_END_CELL]) {
                endCell = cell;
            }
        });

        // 如果有有效的入口和出口，生成路径
        if (startCell && endCell) {
            try {
                maze.findPathBetween(startCell.coords, endCell.coords);
                return true;
            } catch (error) {
                console.log(`⚠️  无法生成解决路径: ${error.message}`);
                return false;
            }
        }

        return false;
    }

    /**
     * 创建带解答的迷宫
     */
    createSolutionMaze(gridConfig, algorithm, seed, canvas, exitConfig) {
        const solutionMaze = buildMaze({
            grid: gridConfig,
            algorithm: algorithm,
            randomSeed: seed,
            element: canvas,
            exitConfig: exitConfig
        });

        return solutionMaze;
    }

    /**
     * 添加入口和出口标记
     */
    addEntranceExitLabels(svgElement, maze) {
        let startCell = null;
        let endCell = null;

        // 查找入口和出口单元格
        maze.forEachCell(cell => {
            if (cell.metadata[METADATA_START_CELL]) {
                startCell = cell;
            }
            if (cell.metadata[METADATA_END_CELL]) {
                endCell = cell;
            }
        });

        // 添加入口标记（绿色圆圈）- 调整到格子中心
        if (startCell && startCell.metadata[METADATA_RAW_COORDS]) {
            const [sx, sy] = startCell.metadata[METADATA_RAW_COORDS];
            let yAdj = sy;
            // if (maze.metadata?.cellShape === SHAPE_TRIANGLE) {
            //     const offset = this.triangleCentroidOffsetPx(startCell);
            //     // 根据出口方向决定向上或向下偏移（使标记更贴近开口一侧）
            //     const exitDir = startCell.metadata[METADATA_START_CELL];
            //     if (exitDir === DIRECTION_SOUTH) {
            //         yAdj = sy + offset;
            //     } else if (exitDir === DIRECTION_NORTH) {
            //         yAdj = sy - offset;
            //     } yAdj+15 ey+10
            // }
            this.addSymbolLabel(svgElement, sx, yAdj+30, '●', '#FF0000', 'bold', 100);
        }

        // 添加出口标记（数学叉号）- 调整到格子中心
        if (endCell && endCell.metadata[METADATA_RAW_COORDS]) {
            const [ex, ey] = endCell.metadata[METADATA_RAW_COORDS];
            this.addSymbolLabel(svgElement, ex, ey+30, '⨯', '#FF0000', 'bold', 100);
        }
    }

        // 三角形：根据上下相邻三角形重心距离，估算重心到底边的偏移（= 两重心距离的一半）
        triangleCentroidOffsetPx(cell) {
            try {
                // 原始屏幕坐标（重心）
                const [cx, cy] = cell.metadata[METADATA_RAW_COORDS] || [];
                if (cx == null || cy == null) return 0;

                const north = (cell.neighbours || {})[DIRECTION_NORTH];
                const south = (cell.neighbours || {})[DIRECTION_SOUTH];

                // 选择一个“上下”邻居（与本三角共享底边），优先北，其次南
                const vertNeighbour = north || south;
                if (!vertNeighbour) return 0;

                const [nx, ny] = (vertNeighbour.metadata || {})[METADATA_RAW_COORDS] || [];
                if (nx == null || ny == null) return 0;

                const dy = Math.abs(ny - cy);

                // 两个相邻（共底边）三角形重心的纵向间距 = 2 * (等边三角高/3) = 2h/3
                // 因此重心到底边的距离 = (2h/3)/2 = h/3 = dy/2
                return dy / 2;
            } catch {
                return 0;
            }
        }
        

    /**
     * 在SVG中添加符号标记（圆圈或叉号）
     */
    addSymbolLabel(svgElement, x, y, symbol, color = '#FF0000', fontWeight = 'bold', fontSize = 40) {
        const textElement = document.createElementNS('http://www.w3.org/2000/svg', 'text');

        textElement.setAttribute('x', x);
        textElement.setAttribute('y', y);
        textElement.setAttribute('text-anchor', 'middle');
        textElement.setAttribute('dominant-baseline', 'middle');
        textElement.setAttribute('fill', color);
        textElement.setAttribute('font-family', 'Arial, sans-serif');
        textElement.setAttribute('font-size', fontSize);
        textElement.setAttribute('font-weight', fontWeight);
        textElement.setAttribute('stroke', 'white');
        textElement.setAttribute('stroke-width', '1');
        textElement.textContent = symbol;

        svgElement.appendChild(textElement);
    }

    /**
     * 在SVG中添加文字标记
     */
    addTextLabel(svgElement, x, y, text, color = '#FF0000', fontWeight = 'bold') {
        const textElement = document.createElementNS('http://www.w3.org/2000/svg', 'text');

        textElement.setAttribute('x', x);
        textElement.setAttribute('y', y);
        textElement.setAttribute('text-anchor', 'middle');
        textElement.setAttribute('dominant-baseline', 'middle');
        textElement.setAttribute('fill', color);
        textElement.setAttribute('font-family', 'Arial, sans-serif');
        textElement.setAttribute('font-size', '30');
        textElement.setAttribute('font-weight', fontWeight);
        textElement.setAttribute('stroke', 'white');
        textElement.setAttribute('stroke-width', '0.5');
        textElement.textContent = text;

        svgElement.appendChild(textElement);
    }

    /**
     * 创建SVG画布
     */
    createSVGCanvas(width = 512, height = 512) {
        const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
        global.document = dom.window.document;
        global.window = dom.window;

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        svg.setAttribute('width', width);
        svg.setAttribute('height', height);
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

        return svg;
    }

    /**
     * 修复SVG中的墙壁宽度为固定值
     */
    fixSvgStrokeWidth(svgElement) {
        // 获取所有有 stroke-width 属性的元素（line 和 path）
        const lines = svgElement.querySelectorAll('line[stroke-width]');
        const paths = svgElement.querySelectorAll('path[stroke-width]');

        // 修改所有 line 元素的 stroke-width
        lines.forEach(line => {
            line.setAttribute('stroke-width', this.wallWidth);
        });

        // 修改所有 path 元素的 stroke-width（如果有 stroke）
        paths.forEach(path => {
            if (path.getAttribute('stroke') && path.getAttribute('stroke') !== 'none') {
                path.setAttribute('stroke-width', this.wallWidth);
            }
        });
    }

    /**
     * 生成单个迷宫
     */
    async generateMaze(config) {
        const {
            shape = SHAPE_SQUARE,
            width = 10,
            height = 10,
            layers = 10,
            algorithm = ALGORITHM_RECURSIVE_BACKTRACK,
            exitConfig = EXITS_VERTICAL,
            seed = null,
            filename = null
        } = config;

        console.log(`生成迷宫: ${shape}, 算法: ${algorithm}, 尺寸: ${width}x${height || layers}`);

        // 根据迷宫配置计算图片尺寸
        const imageSize = this.calculateImageSize(shape, width, height, layers);
        console.log(`图片尺寸: ${imageSize.width}x${imageSize.height}px (格子大小: ${this.cellSize}px)`);

        // 创建两个SVG画布：一个用于迷宫，一个用于带解答的迷宫
        const mazeCanvas = this.createSVGCanvas(imageSize.width, imageSize.height);
        const solutionCanvas = this.createSVGCanvas(imageSize.width, imageSize.height);

        // 准备网格配置
        const gridConfig = { cellShape: shape };
        if (shape === SHAPE_CIRCLE) {
            gridConfig.layers = layers;
        } else {
            gridConfig.width = width;
            gridConfig.height = height;
        }

        // 使用确定的种子
        const mazeSeed = seed || Date.now();

        // 生成迷宫
        const maze = buildMaze({
            grid: gridConfig,
            algorithm: algorithm,
            randomSeed: mazeSeed,
            element: mazeCanvas,
            exitConfig: exitConfig
        });

        // 初始化并运行算法
        maze.initialise();
        maze.runAlgorithm.toCompletion();
        
        // const startCell = maze.getCellByCoordinates(0, Math.floor(maze.metadata.height/2)); // 最左中点
        // const endCell   = maze.getCellByCoordinates(maze.metadata.width-1, Math.floor(maze.metadata.height/2)); // 最右中点
        // if (startCell) startCell.metadata[METADATA_START_CELL] = DIRECTION_WEST; // 向左打洞
        // if (endCell)   endCell.metadata[METADATA_END_CELL]   = DIRECTION_EAST;  // 向右打洞

        // 渲染迷宫（无解答）
        maze.render();

        // 先保存无标记版本
        const mazeSvgNoMarkers = mazeCanvas.cloneNode(true);

        // 然后添加标记
        this.addEntranceExitLabels(mazeCanvas, maze);

        // 生成解决路径
        const hasValidExits = this.generateSolution(maze);

        // 计算“富转向”难度
        let difficultyInfo = null;
        if (hasValidExits) {
            const extra = {};
            if (shape === SHAPE_CIRCLE) {
                // 从你的 maze/grid 中拿每层扇区数（下面是示例，按你实际实现改）
                // 假设 maze.grid.layers[i].segments 存有第 i 层分段数：
                extra.segmentsAtLayer = (maze.grid?.layers || []).map(l => l.segments);
            }
            difficultyInfo = this.calculateTurnDifficultyRich(maze, shape, extra);

            const difficultyLevel = this.getDifficultyLevel(difficultyInfo.difficulty);
            console.log(`📊 转向复杂度:`);
            console.log(`   难度等级: ${difficultyLevel} (${(difficultyInfo.difficulty * 100).toFixed(1)}%)`);
            console.log(`   H_turn: ${difficultyInfo.H_turn.toFixed(3)} (norm ${(difficultyInfo.H_turn_norm*100).toFixed(1)}%)`);
            console.log(`   H_rate: ${difficultyInfo.H_rate.toFixed(3)} (norm ${(difficultyInfo.H_rate_norm*100).toFixed(1)}%)`);
            console.log(`   步数: ${difficultyInfo.steps}, 转向统计: ${JSON.stringify(difficultyInfo.turnCounts)}`);
        }


        // 渲染带解答的迷宫 - 使用相同种子确保相同迷宫结构
        let solutionMaze = null;
        if (hasValidExits) {
            solutionMaze = this.createSolutionMaze(gridConfig, algorithm, mazeSeed, solutionCanvas, exitConfig);
            solutionMaze.initialise();
            solutionMaze.runAlgorithm.toCompletion();

            // 设置相同的入口和出口
            // const solutionStartCell = solutionMaze.getCellByCoordinates(0, Math.floor(solutionMaze.metadata.height/2));
            // const solutionEndCell = solutionMaze.getCellByCoordinates(solutionMaze.metadata.width-1, Math.floor(solutionMaze.metadata.height/2));
            // if (solutionStartCell) solutionStartCell.metadata[METADATA_START_CELL] = DIRECTION_WEST;
            // if (solutionEndCell) solutionEndCell.metadata[METADATA_END_CELL] = DIRECTION_EAST;

            // 为解答迷宫生成相同的路径
            this.generateSolution(solutionMaze);
            solutionMaze.render();

            // 先保存无标记版本
            const solutionCanvasNoMarkers = solutionCanvas.cloneNode(true);

            // 然后添加标记
            this.addEntranceExitLabels(solutionCanvas, solutionMaze);
        }

        // 生成文件名
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const difficultyLevel = difficultyInfo ? this.getDifficultyLevel(difficultyInfo.difficulty) : 'UNKNOWN';
        const difficultyScore = difficultyInfo ? Math.round(difficultyInfo.difficulty * 100) : 0;

        const baseName = filename ? filename.replace(/\.(svg|png)$/, '') :
            `maze_${shape}_${algorithm}_${width}x${height || layers}_${difficultyLevel}_${difficultyScore}_${timestamp}`;

        // 保存迷宫PNG文件
        const mazeFilename = `${baseName}.png`;
        const mazeFilepath = path.join(this.outputDir, mazeFilename);

        const mazeConvertSuccess = await this.convertSvgToPng(
            mazeCanvas.outerHTML,
            mazeFilepath,
            imageSize.width,
            imageSize.height
        );
        if (mazeConvertSuccess) {
            console.log(`✓ 迷宫已保存: ${mazeFilepath}`);
        } else {
            console.log(`❌ 迷宫PNG转换失败，保存SVG版本`);
            const svgPath = mazeFilepath.replace('.png', '.svg');
            fs.writeFileSync(svgPath, mazeCanvas.outerHTML);
            console.log(`✓ 迷宫SVG已保存: ${svgPath}`);
        }

        // 保存无标记版本的迷宫PNG文件
        const mazeNoMarkersFilepath = path.join(this.noMarkersDir, mazeFilename);
        const mazeNoMarkersConvertSuccess = await this.convertSvgToPng(
            mazeSvgNoMarkers.outerHTML,
            mazeNoMarkersFilepath,
            imageSize.width,
            imageSize.height
        );
        if (mazeNoMarkersConvertSuccess) {
            console.log(`✓ 无标记迷宫已保存: ${mazeNoMarkersFilepath}`);
        } else {
            console.log(`❌ 无标记迷宫PNG转换失败，保存SVG版本`);
            const svgPath = mazeNoMarkersFilepath.replace('.png', '.svg');
            fs.writeFileSync(svgPath, mazeSvgNoMarkers.outerHTML);
            console.log(`✓ 无标记迷宫SVG已保存: ${svgPath}`);
        }

        // 保存解答PNG文件
        let solutionFilepath = null;
        if (hasValidExits) {
            const solutionFilename = `${baseName}_solution.png`;
            solutionFilepath = path.join(this.solutionDir, solutionFilename);

            const solutionConvertSuccess = await this.convertSvgToPng(
                solutionCanvas.outerHTML,
                solutionFilepath,
                imageSize.width,
                imageSize.height
            );
            if (solutionConvertSuccess) {
                console.log(`✓ 解答已保存: ${solutionFilepath}`);
            } else {
                console.log(`❌ 解答PNG转换失败，保存SVG版本`);
                const svgPath = solutionFilepath.replace('.png', '.svg');
                fs.writeFileSync(svgPath, solutionCanvas.outerHTML);
                console.log(`✓ 解答SVG已保存: ${svgPath}`);
            }

            // 保存无标记版本的解答PNG文件
            // const solutionNoMarkersFilepath = path.join(this.noMarkersDir, solutionFilename);
            // const solutionNoMarkersConvertSuccess = await this.convertSvgToPng(solutionCanvasNoMarkers.outerHTML, solutionNoMarkersFilepath);
            // if (solutionNoMarkersConvertSuccess) {
            //     console.log(`✓ 无标记解答已保存: ${solutionNoMarkersFilepath}`);
            // } else {
            //     console.log(`❌ 无标记解答PNG转换失败，保存SVG版本`);
            //     const svgPath = solutionNoMarkersFilepath.replace('.png', '.svg');
            //     fs.writeFileSync(svgPath, solutionCanvasNoMarkers.outerHTML);
            //     console.log(`✓ 无标记解答SVG已保存: ${svgPath}`);
            // }
        } else {
            console.log(`⚠️  没有生成解答 (没有有效的出入口)`);
        }

        // 生成并保存训练数据集文件（mask、cell map、元数据）
        if (hasValidExits) {
            console.log(`📦 生成训练数据集文件...`);

            // a. 提取路径格子坐标
            const pathCoords = maze.metadata[METADATA_PATH];

            // 获取起点终点坐标
            let startCellCoords = null;
            let endCellCoords = null;
            maze.forEachCell(cell => {
                if (cell.metadata[METADATA_START_CELL]) {
                    startCellCoords = cell.coords;
                }
                if (cell.metadata[METADATA_END_CELL]) {
                    endCellCoords = cell.coords;
                }
            });

            // 构建格子ID列表（用于训练时计算交集）
            const pathCellIds = [];
            pathCoords.forEach(coords => {
                const cellId = this.coordsToCellId(coords, maze, shape);
                pathCellIds.push(cellId);
            });

            // b. 生成格子分割图PNG（RGB编码）
            console.log(`  生成格子分割图...`);
            const cellMap = generateCellSegmentationMap(maze, imageSize.width, imageSize.height);
            const cellMapFilepath = path.join(this.metadataDir, `${baseName}_cell_map.png`);
            const cellMapSaveSuccess = await this.saveCellMapAsPNG(cellMap, imageSize.width, imageSize.height, cellMapFilepath);
            if (cellMapSaveSuccess) {
                console.log(`✓ 格子分割图已保存: ${cellMapFilepath}`);
            } else {
                console.log(`❌ 格子分割图保存失败`);
            }

            // c. 生成GT路径mask PNG（直接基于path_cell_ids和cellMap生成）
            console.log(`  生成路径mask...`);
            const pathCellIdsSet = new Set(pathCellIds);
            const pathMask = new Uint8Array(imageSize.width * imageSize.height);
            for (let i = 0; i < cellMap.length; i++) {
                // 如果该像素所属的cell ID在pathCellIds中，则设为1，否则设为0
                pathMask[i] = pathCellIdsSet.has(cellMap[i]) ? 1 : 0;
            }
            const pathMaskFilepath = path.join(this.metadataDir, `${baseName}_path_mask.png`);
            const maskSaveSuccess = await this.saveMaskAsPNG(pathMask, pathMaskFilepath, imageSize.width, imageSize.height);
            if (maskSaveSuccess) {
                console.log(`✓ 路径mask已保存: ${pathMaskFilepath}`);
            } else {
                console.log(`❌ 路径mask保存失败`);
            }

            // d. 保存元数据JSON
            console.log(`  保存元数据...`);

            const metadata = {
                path_coordinates: pathCoords,
                path_cell_ids: pathCellIds,
                start_cell: startCellCoords,
                end_cell: endCellCoords,
                maze_config: {
                    shape: shape,
                    width: width,
                    height: height,
                    layers: layers,
                    algorithm: algorithm,
                    seed: mazeSeed
                },
                image_size: {
                    width: imageSize.width,
                    height: imageSize.height,
                    cell_size: this.cellSize,
                    wall_width: this.wallWidth
                },
                difficulty: difficultyInfo
            };

            const metadataFilepath = path.join(this.metadataDir, `${baseName}.json`);
            fs.writeFileSync(metadataFilepath, JSON.stringify(metadata, null, 2));
            console.log(`✓ 元数据已保存: ${metadataFilepath}`);
        }

        // 清理资源
        maze.dispose();
        if (solutionMaze) {
            solutionMaze.dispose();
        }

        return {
            mazeFile: mazeFilepath,
            solutionFile: solutionFilepath,
            difficulty: difficultyInfo
        };
    }

    /**
     * 批量生成迷宫
     */
    async generateBatch(configs) {
        console.log(`开始批量生成 ${configs.length} 个迷宫...`);
        const results = [];

        for (let i = 0; i < configs.length; i++) {
            console.log(`\n进度: ${i + 1}/${configs.length}`);
            try {
                const result = await this.generateMaze(configs[i]);
                results.push({
                    success: true,
                    mazeFile: result.mazeFile,
                    solutionFile: result.solutionFile,
                    config: configs[i],
                    difficulty: result.difficulty
                });
            } catch (error) {
                console.error(`❌ 生成失败:`, error.message);
                results.push({ success: false, error: error.message, config: configs[i] });
            }
        }

        // 输出统计信息
        const successful = results.filter(r => r.success).length;
        const failed = results.length - successful;
        const withSolutions = results.filter(r => r.success && r.solutionFile).length;

        // 难度统计
        const successfulWithDifficulty = results.filter(r => r.success && r.difficulty);
        const difficultyStats = {};
        let totalDifficulty = 0;
        let difficultyCount = 0;

        successfulWithDifficulty.forEach(result => {
            if (result.difficulty) {
                const level = this.getDifficultyLevel(result.difficulty.difficulty);
                difficultyStats[level] = (difficultyStats[level] || 0) + 1;
                totalDifficulty += result.difficulty.difficulty;
                difficultyCount++;
            }
        });

        const averageDifficulty = difficultyCount > 0 ? totalDifficulty / difficultyCount : 0;

        console.log(`\n📊 批量生成完成:`);
        console.log(`  ✓ 成功: ${successful}`);
        console.log(`  ❌ 失败: ${failed}`);
        console.log(`  🔍 包含解答: ${withSolutions}`);
        console.log(`  📁 带标记迷宫目录: ${this.outputDir}`);
        console.log(`  📁 无标记迷宫目录: ${this.noMarkersDir}`);
        console.log(`  📁 解答目录: ${this.solutionDir}`);

        if (difficultyCount > 0) {
            console.log(`\n🎯 难度统计:`);
            console.log(`  平均难度: ${(averageDifficulty * 100).toFixed(1)}%`);
            Object.entries(difficultyStats).forEach(([level, count]) => {
                console.log(`  ${level}: ${count} 个`);
            });
        }

        return results;
    }

    /**
     * 从配置文件生成
     */
    async generateFromConfig(configFile) {
        if (!fs.existsSync(configFile)) {
            throw new Error(`配置文件不存在: ${configFile}`);
        }

        const configData = JSON.parse(fs.readFileSync(configFile, 'utf8'));
        return await this.generateBatch(configData.mazes || [configData]);
    }

    /**
     * 生成预设样例
     */
    async generateSamples() {
        const samples = [
            // 方形迷宫样例
            {
                shape: SHAPE_SQUARE,
                width: 15,
                height: 15,
                algorithm: ALGORITHM_RECURSIVE_BACKTRACK,
                exitConfig: EXITS_VERTICAL,
                filename: 'sample_square_recursive.png'
            },
            {
                shape: SHAPE_SQUARE,
                width: 20,
                height: 15,
                algorithm: ALGORITHM_KRUSKAL,
                exitConfig: EXITS_HORIZONTAL,
                filename: 'sample_square_kruskal.png'
            },
            // 三角形迷宫样例
            {
                shape: SHAPE_TRIANGLE,
                width: 20,
                height: 12,
                algorithm: ALGORITHM_ALDOUS_BRODER,
                exitConfig: EXITS_VERTICAL,
                filename: 'sample_triangle_aldous.png'
            },
            // 六边形迷宫样例
            {
                shape: SHAPE_HEXAGON,
                width: 12,
                height: 10,
                algorithm: ALGORITHM_WILSON,
                exitConfig: EXITS_HARDEST,
                filename: 'sample_hexagon_wilson.png'
            },
            // 圆形迷宫样例
            {
                shape: SHAPE_CIRCLE,
                layers: 8,
                algorithm: ALGORITHM_HUNT_AND_KILL,
                exitConfig: EXITS_HARDEST,
                filename: 'sample_circle_hunt_kill.png'
            }
        ];

        return await this.generateBatch(samples);
    }
}

// 获取所有可用的算法和形状
function getAvailableOptions() {
    return {
        shapes: [SHAPE_SQUARE, SHAPE_TRIANGLE, SHAPE_HEXAGON, SHAPE_CIRCLE],
        algorithms: Object.keys(algorithms).filter(alg => alg !== 'none'),
        exitConfigs: [EXITS_NONE, EXITS_HORIZONTAL, EXITS_VERTICAL, EXITS_HARDEST]
    };
}

// 命令行接口
async function main() {
    const args = process.argv.slice(2);
    const generator = new BatchMazeGenerator();

    if (args.length === 0) {
        console.log(`
🏃‍♂️ 批量迷宫生成器

用法:
  node batch-maze-generator.js samples                    # 生成预设样例
  node batch-maze-generator.js config <file.json>         # 从配置文件生成
  node batch-maze-generator.js single [options]           # 生成单个迷宫
  node batch-maze-generator.js list                       # 列出所有可用选项

单个迷宫选项:
  --shape <shape>        形状: square, triangle, hexagon, circle
  --width <width>        宽度 (方形/三角形/六边形)
  --height <height>      高度 (方形/三角形/六边形)
  --layers <layers>      层数 (圆形)
  --algorithm <alg>      算法: recursiveBacktrack, kruskal, wilson, 等
  --exits <exits>        出入口: vertical, horizontal, hardest, 无
  --seed <seed>          随机种子
  --filename <name>      输出文件名

示例:
  node batch-maze-generator.js single --shape square --width 20 --height 15 --algorithm kruskal
  node batch-maze-generator.js single --shape circle --layers 10 --algorithm wilson --exits hardest
        `);
        return;
    }

    const command = args[0];

    switch (command) {
        case 'samples':
            await generator.generateSamples();
            break;

        case 'config':
            if (args.length < 2) {
                console.error('❌ 请指定配置文件');
                return;
            }
            await generator.generateFromConfig(args[1]);
            break;

        case 'single':
            const config = parseSingleMazeArgs(args.slice(1));
            await generator.generateMaze(config);
            break;

        case 'list':
            const options = getAvailableOptions();
            console.log('🎛️  可用选项:');
            console.log('形状:', options.shapes.join(', '));
            console.log('算法:', options.algorithms.join(', '));
            console.log('出入口:', options.exitConfigs.join(', '));
            break;

        default:
            console.error('❌ 未知命令:', command);
    }
}

function parseSingleMazeArgs(args) {
    const config = {};

    for (let i = 0; i < args.length; i += 2) {
        const key = args[i]?.replace('--', '');
        const value = args[i + 1];

        if (!key || !value) continue;

        switch (key) {
            case 'width':
            case 'height':
            case 'layers':
            case 'seed':
                config[key] = parseInt(value);
                break;
            default:
                config[key] = value;
        }
    }

    return config;
}

// 如果直接运行此脚本
if (import.meta.url === pathToFileURL(process.argv[1]).href) {
    main().catch(console.error);
}

export { BatchMazeGenerator, getAvailableOptions };
// export { BatchMazeGenerator, getAvailableOptions };