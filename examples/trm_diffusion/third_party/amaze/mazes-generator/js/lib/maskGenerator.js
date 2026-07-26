import {
    SHAPE_SQUARE, SHAPE_TRIANGLE, SHAPE_HEXAGON, SHAPE_CIRCLE,
    METADATA_RAW_COORDS, METADATA_END_CELL
} from './constants.js';

/**
 * 生成基于格子形状的二进制掩码
 * @param {Object} maze - 迷宫对象
 * @param {Set} solutionCells - 解空间格子的坐标集合
 * @param {number} imageWidth - 输出图像宽度
 * @param {number} imageHeight - 输出图像高度
 * @returns {Uint8Array} 二进制掩码数组 (1=解空间, 0=其他)
 */
export function generateShapeAwareMask(maze, solutionCells, imageWidth = 512, imageHeight = 512) {
    const maskData = new Uint8Array(imageWidth * imageHeight);

    // 确定网格形状 - 从不同的可能位置获取
    let shape;
    if (maze.metadata.cellShape) {
        shape = maze.metadata.cellShape;
    } else if (maze.metadata.grid && maze.metadata.grid.cellShape) {
        shape = maze.metadata.grid.cellShape;
    } else {
        // 根据网格类型推断形状
        if (maze.isSquare === true) {
            shape = SHAPE_SQUARE;
        } else if (maze.isSquare === false) {
            // 需要区分三角形和六边形网格
            // 检查网格的特征来判断
            if (maze.metadata.layers !== undefined) {
                shape = SHAPE_CIRCLE;
            } else {
                // 检查网格中心的格子邻居数量来区分三角形和六边形
                let testCell = maze.getCellByCoordinates(1, 1); // 尝试中心格子
                if (!testCell) {
                    testCell = maze.getCellByCoordinates(0, 0); // 备选
                }

                if (testCell) {
                    const neighborCount = testCell.neighbours.toArray().length;
                    if (neighborCount <= 3) {
                        shape = SHAPE_TRIANGLE;
                    } else {
                        shape = SHAPE_HEXAGON;
                    }
                } else {
                    shape = SHAPE_TRIANGLE; // 默认
                }
            }
        } else if (maze.metadata.layers !== undefined) {
            shape = SHAPE_CIRCLE;
        } else {
            shape = SHAPE_SQUARE; // 默认
        }
    }

    // 特殊处理圆形格子的连通性
    let extendedSolutionCells = solutionCells;
    if (shape === SHAPE_CIRCLE && solutionCells.size > 1) {
        extendedSolutionCells = new Set(solutionCells);

        // 检查是否有中心格子和外层格子
        const cellCoords = Array.from(solutionCells).map(coordsStr => coordsStr.split(',').map(Number));
        const hasCenterCell = cellCoords.some(coords => coords[0] === 0);
        const hasOuterCells = cellCoords.some(coords => coords[0] > 0);

        if (hasCenterCell && hasOuterCells) {
            // 添加连接路径：从中心到第一层的所有相关扇形
            const outerCells = cellCoords.filter(coords => coords[0] > 0);
            outerCells.forEach(([layer, cellIndex]) => {
                if (layer === 1) {
                    // 第一层直接连接到中心，已经有连通性
                    return;
                }
                // 对于更外层，需要添加连接路径
                for (let l = 1; l < layer; l++) {
                    // 添加从内层到外层的连接路径格子
                    const cellCounts = cellCountsForLayers(maze.metadata.layers);
                    const outerLayerCells = cellCounts[layer];
                    const innerLayerCells = cellCounts[l];
                    const cellsPerInnerCell = outerLayerCells / innerLayerCells;
                    const correspondingInnerCell = Math.floor(cellIndex / cellsPerInnerCell);
                    extendedSolutionCells.add(`${l},${correspondingInnerCell}`);
                }
            });
        }
    }

    // 遍历每个解空间格子，将其对应的像素区域设为1
    extendedSolutionCells.forEach(coordsStr => {
        const coords = coordsStr.split(',').map(Number);
        const cell = maze.getCellByCoordinates(...coords);

        if (cell && cell.metadata[METADATA_RAW_COORDS]) {
            const [centerX, centerY] = cell.metadata[METADATA_RAW_COORDS];
            fillCellMask(maskData, imageWidth, imageHeight, centerX, centerY, coords, shape, maze);
        }
    });

    // 特殊处理：连接终点格子和其相邻的解空间格子
    if (shape === SHAPE_CIRCLE) {
        extendedSolutionCells.forEach(coordsStr => {
            const coords = coordsStr.split(',').map(Number);
            const cell = maze.getCellByCoordinates(...coords);

            if (cell && cell.metadata[METADATA_END_CELL]) {
                // 找到终点格子，检查其邻居
                cell.neighbours.toArray().forEach(neighbor => {
                    const neighborCoordStr = neighbor.coords.join(',');
                    if (extendedSolutionCells.has(neighborCoordStr)) {
                        // 这个邻居也在解空间中，需要连接
                        connectCircleCells(maskData, imageWidth, imageHeight, coords, neighbor.coords, maze);
                    }
                });
            }
        });
    }

    return maskData;
}

/**
 * 连接两个圆形格子，填充它们之间的整个连通区域
 */
function connectCircleCells(maskData, imageWidth, imageHeight, coords1, coords2, maze) {
    const layers = maze.metadata.layers;
    const cellCounts = cellCountsForLayers(layers);

    // 使用与绘制函数相同的缩放计算
    const requiredWidth = layers * 2;
    const requiredHeight = layers * 2;
    const shapeSpecificLineWidthAdjustment = 1.5;

    const GLOBAL_LINE_WIDTH_ADJUSTMENT = 0.1;
    const verticalLineWidth = imageHeight * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredHeight;
    const horizontalLineWidth = imageWidth * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredWidth;
    const lineWidth = Math.min(verticalLineWidth, horizontalLineWidth);

    const magnification = Math.min((imageWidth - lineWidth)/requiredWidth, (imageHeight - lineWidth)/requiredHeight);
    const xOffset = lineWidth / 2;
    const yOffset = lineWidth / 2;

    function xCoord(x) {
        return xOffset + x * magnification;
    }
    function yCoord(y) {
        return yOffset + y * magnification;
    }

    const cx = layers;
    const cy = layers;

    function polarToXy(angle, distance) {
        return [cx + distance * Math.sin(angle), cy - distance * Math.cos(angle)];
    }

    function getCellCoords(l, c) {
        const cellsInLayer = cellCounts[l],
            anglePerCell = Math.PI * 2 / cellsInLayer,
            startAngle = anglePerCell * c,
            endAngle = startAngle + anglePerCell,
            innerDistance = l,
            outerDistance = l + 1;

        return [startAngle, endAngle, innerDistance, outerDistance];
    }

    // 如果其中一个是中心格子，另一个是外层格子，填充连接的扇形区域
    const [layer1, cellIndex1] = coords1;
    const [layer2, cellIndex2] = coords2;

    if ((layer1 === 0 && layer2 > 0) || (layer2 === 0 && layer1 > 0)) {
        // 中心格子连接到外层格子，填充扇形连接区域
        const outerLayer = layer1 === 0 ? layer2 : layer1;
        const outerCellIndex = layer1 === 0 ? cellIndex2 : cellIndex1;

        const [startAngle, endAngle, innerDistance, outerDistance] = getCellCoords(outerLayer, outerCellIndex);

        // 填充从中心到外层格子的扇形连接区域
        for (let distance = 0; distance <= outerLayer; distance += 0.1) {
            const numSteps = Math.max(50, Math.ceil((endAngle - startAngle) * distance * 50));
            for (let i = 0; i <= numSteps; i++) {
                const angle = startAngle + (endAngle - startAngle) * i / numSteps;
                const [x, y] = polarToXy(angle, distance);
                const px = Math.round(xCoord(x));
                const py = Math.round(yCoord(y));

                // 填充周围的像素区域
                for (let dx = -2; dx <= 2; dx++) {
                    for (let dy = -2; dy <= 2; dy++) {
                        const finalX = px + dx;
                        const finalY = py + dy;
                        if (finalX >= 0 && finalX < imageWidth && finalY >= 0 && finalY < imageHeight) {
                            const maskIndex = finalY * imageWidth + finalX;
                            maskData[maskIndex] = 1;
                        }
                    }
                }
            }
        }
    } else if (layer1 === layer2) {
        // 同一层的相邻格子，填充它们之间的扇形区域
        const layer = layer1;
        const [startAngle1, endAngle1, innerDistance, outerDistance] = getCellCoords(layer, cellIndex1);
        const [startAngle2, endAngle2] = getCellCoords(layer, cellIndex2);

        const minAngle = Math.min(startAngle1, startAngle2);
        const maxAngle = Math.max(endAngle1, endAngle2);

        // 填充两个格子之间的扇形区域
        for (let distance = innerDistance; distance <= outerDistance; distance += 0.1) {
            const numSteps = Math.max(50, Math.ceil((maxAngle - minAngle) * distance * 50));
            for (let i = 0; i <= numSteps; i++) {
                const angle = minAngle + (maxAngle - minAngle) * i / numSteps;
                const [x, y] = polarToXy(angle, distance);
                const px = Math.round(xCoord(x));
                const py = Math.round(yCoord(y));

                // 填充周围的像素区域
                for (let dx = -1; dx <= 1; dx++) {
                    for (let dy = -1; dy <= 1; dy++) {
                        const finalX = px + dx;
                        const finalY = py + dy;
                        if (finalX >= 0 && finalX < imageWidth && finalY >= 0 && finalY < imageHeight) {
                            const maskIndex = finalY * imageWidth + finalX;
                            maskData[maskIndex] = 1;
                        }
                    }
                }
            }
        }
    }
}

/**
 * 根据形状填充格子掩码
 */
function fillCellMask(maskData, imageWidth, imageHeight, centerX, centerY, coords, shape, maze) {
    // 忽略 centerX, centerY（这些是显示坐标系的坐标）
    // 直接使用 coords 和 maze 进行坐标变换
    switch (shape) {
        case SHAPE_SQUARE:
            fillSquareMask(maskData, imageWidth, imageHeight, coords, maze);
            break;
        case SHAPE_TRIANGLE:
            fillTriangleMask(maskData, imageWidth, imageHeight, coords, maze);
            break;
        case SHAPE_HEXAGON:
            fillHexagonMask(maskData, imageWidth, imageHeight, coords, maze);
            break;
        case SHAPE_CIRCLE:
            fillCircleMask(maskData, imageWidth, imageHeight, coords, maze);
            break;
        default:
            fillSquareMask(maskData, imageWidth, imageHeight, coords, maze);
    }
}

/**
 * 填充方形格子掩码
 */
function fillSquareMask(maskData, imageWidth, imageHeight, coords, maze) {
    const [x, y] = coords;
    const gridWidth = maze.metadata.width;
    const gridHeight = maze.metadata.height;

    const cellPixelWidth = Math.max(1, Math.floor(imageWidth / gridWidth));
    const cellPixelHeight = Math.max(1, Math.floor(imageHeight / gridHeight));

    const halfWidth = Math.floor(cellPixelWidth / 2);
    const halfHeight = Math.floor(cellPixelHeight / 2);

    // 计算格子中心在图像中的位置
    const centerX = (x + 0.5) * cellPixelWidth;
    const centerY = (y + 0.5) * cellPixelHeight;

    const startX = Math.max(0, Math.floor(centerX) - halfWidth);
    const endX = Math.min(imageWidth - 1, Math.floor(centerX) + halfWidth);
    const startY = Math.max(0, Math.floor(centerY) - halfHeight);
    const endY = Math.min(imageHeight - 1, Math.floor(centerY) + halfHeight);

    for (let py = startY; py <= endY; py++) {
        for (let px = startX; px <= endX; px++) {
            const index = py * imageWidth + px;
            maskData[index] = 1;
        }
    }
}

/**
 * 填充三角形格子掩码（与 maze.js buildTriangularGrid 的渲染函数完全一致）
 *
 * 关键修复：
 * 1. 使用与 maze.js:528 完全相同的空间需求计算和线宽调整参数 (0.8)
 * 2. 使用更精确的重心坐标法 isPointInTriangleBarycentric()
 * 3. 使用像素中心点 (px + 0.5, py + 0.5) 进行判断，提高边界精度
 * 4. 扩大边界框以确保不漏掉边界像素
 */
function fillTriangleMask(maskData, imageWidth, imageHeight, coords, maze) {
    const [x, y] = coords;
    const verticalAltitude = Math.sin(Math.PI/3);

    // 使用与绘制函数相同的 hasBaseOnSouthSide 函数
    function hasBaseOnSouthSide(x, y) {
        return (x + y) % 2;
    }

    // 使用与绘制函数相同的 getCornerCoords 函数
    function getCornerCoords(x, y) {
        let p1x, p1y, p2x, p2y, p3x, p3y;

        if (hasBaseOnSouthSide(x, y)) {
            p1x = x/2;
            p1y = (y+1) * verticalAltitude;
            p2x = (x+1)/2;
            p2y = p1y - verticalAltitude;
            p3x = p1x + 1;
            p3y = p1y;
        } else {
            p1x = x/2;
            p1y = y * verticalAltitude;
            p2x = (x+1)/2;
            p2y = p1y + verticalAltitude;
            p3x = p1x + 1;
            p3y = p1y;
        }

        return [p1x, p1y, p2x, p2y, p3x, p3y];
    }

    // 获取三角形的原始坐标
    const [p1x, p1y, p2x, p2y, p3x, p3y] = getCornerCoords(x, y);

    // 使用与绘制函数相同的缩放计算
    const requiredWidth = 0.5 + maze.metadata.width/2;
    const requiredHeight = maze.metadata.height * verticalAltitude;
    const shapeSpecificLineWidthAdjustment = 0.8;

    const GLOBAL_LINE_WIDTH_ADJUSTMENT = 0.1;
    const verticalLineWidth = imageHeight * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredHeight;
    const horizontalLineWidth = imageWidth * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredWidth;
    const lineWidth = Math.min(verticalLineWidth, horizontalLineWidth);

    const magnification = Math.min((imageWidth - lineWidth)/requiredWidth, (imageHeight - lineWidth)/requiredHeight);
    const xOffset = lineWidth / 2;
    const yOffset = lineWidth / 2;

    // 转换顶点坐标到图像坐标系（使用与绘制函数相同的转换）
    function xCoord(x) {
        return xOffset + x * magnification;
    }
    function yCoord(y) {
        return yOffset + y * magnification;
    }

    const tp1x = xCoord(p1x);
    const tp1y = yCoord(p1y);
    const tp2x = xCoord(p2x);
    const tp2y = yCoord(p2y);
    const tp3x = xCoord(p3x);
    const tp3y = yCoord(p3y);

    // 计算边界框，不被图像边界截断
    const minX = Math.floor(Math.min(tp1x, tp2x, tp3x) - 0.5);
    const maxX = Math.ceil(Math.max(tp1x, tp2x, tp3x) + 0.5);
    const minY = Math.floor(Math.min(tp1y, tp2y, tp3y) - 0.5);
    const maxY = Math.ceil(Math.max(tp1y, tp2y, tp3y) + 0.5);

    // 填充三角形内的像素，只对有效像素区域进行填充
    for (let py = Math.max(0, minY); py <= Math.min(imageHeight - 1, maxY); py++) {
        for (let px = Math.max(0, minX); px <= Math.min(imageWidth - 1, maxX); px++) {
            // 使用像素中心点 (px + 0.5, py + 0.5) 进行判断
            const pixelCenterX = px + 0.5;
            const pixelCenterY = py + 0.5;

            if (isPointInTriangleBarycentric(pixelCenterX, pixelCenterY, tp1x, tp1y, tp2x, tp2y, tp3x, tp3y)) {
                const index = py * imageWidth + px;
                maskData[index] = 1;
            }
        }
    }
}

/**
 * 填充六边形格子掩码（使用与绘制函数一致的几何计算）
 */
function fillHexagonMask(maskData, imageWidth, imageHeight, coords, maze) {
    const [x, y] = coords;
    const yOffset1 = Math.cos(Math.PI / 3);
    const yOffset2 = 2 - yOffset1;
    const yOffset3 = 2;
    const xOffset = Math.sin(Math.PI / 3);

    // 使用与绘制函数相同的缩放计算
    const requiredWidth = maze.metadata.width * 2 * xOffset + Math.min(1, maze.metadata.height - 1) * xOffset;
    const requiredHeight = maze.metadata.height * yOffset2 + yOffset1;
    const shapeSpecificLineWidthAdjustment = 1.5;

    const GLOBAL_LINE_WIDTH_ADJUSTMENT = 0.1;
    const verticalLineWidth = imageHeight * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredHeight;
    const horizontalLineWidth = imageWidth * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredWidth;
    const lineWidth = Math.min(verticalLineWidth, horizontalLineWidth);

    const magnification = Math.min((imageWidth - lineWidth)/requiredWidth, (imageHeight - lineWidth)/requiredHeight);
    const xOffsetDrawing = lineWidth / 2;
    const yOffsetDrawing = lineWidth / 2;

    // 转换顶点坐标到图像坐标系（使用与绘制函数相同的转换）
    function xCoord(x) {
        return xOffsetDrawing + x * magnification;
    }
    function yCoord(y) {
        return yOffsetDrawing + y * magnification;
    }

    // 使用与绘制函数相同的 getCornerCoords 函数
    function getCornerCoords(x, y) {
        const rowXOffset = Math.abs(y % 2) * xOffset,
            p1x = rowXOffset + x * xOffset * 2,
            p1y = yOffset1 + y * yOffset2,
            p2x = p1x,
            p2y = (y + 1) * yOffset2,
            p3x = rowXOffset + (2 * x + 1) * xOffset,
            p3y = y * yOffset2 + yOffset3,
            p4x = p2x + 2 * xOffset,
            p4y = p2y,
            p5x = p4x,
            p5y = p1y,
            p6x = p3x,
            p6y = y * yOffset2;

        return [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y];
    }

    // 获取六边形的原始坐标
    const [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y] = getCornerCoords(x, y);

    // 将六边形顶点转换到图像坐标系
    const vertices = [
        [xCoord(p1x), yCoord(p1y)],
        [xCoord(p2x), yCoord(p2y)],
        [xCoord(p3x), yCoord(p3y)],
        [xCoord(p4x), yCoord(p4y)],
        [xCoord(p5x), yCoord(p5y)],
        [xCoord(p6x), yCoord(p6y)]
    ];

    // 计算边界框，不被图像边界截断
    const xs = vertices.map(v => v[0]);
    const ys = vertices.map(v => v[1]);
    const minX = Math.floor(Math.min(...xs) - 0.5);
    const maxX = Math.ceil(Math.max(...xs) + 0.5);
    const minY = Math.floor(Math.min(...ys) - 0.5);
    const maxY = Math.ceil(Math.max(...ys) + 0.5);

    // 填充六边形内的像素，只对有效像素区域进行填充
    for (let py = Math.max(0, minY); py <= Math.min(imageHeight - 1, maxY); py++) {
        for (let px = Math.max(0, minX); px <= Math.min(imageWidth - 1, maxX); px++) {
            // 使用像素中心点进行判断
            if (isPointInPolygon(px + 0.5, py + 0.5, vertices)) {
                const index = py * imageWidth + px;
                maskData[index] = 1;
            }
        }
    }
}

/**
 * 填充圆形格子掩码（使用与绘制函数一致的几何计算）
 */
function fillCircleMask(maskData, imageWidth, imageHeight, coords, maze) {
    const [layer, cellIndex] = coords;
    const layers = maze.metadata.layers;

    // 使用与绘制函数相同的 cellCounts 计算逻辑
    const cellCounts = cellCountsForLayers(layers);

    // 使用与绘制函数相同的缩放计算
    const requiredWidth = layers * 2;
    const requiredHeight = layers * 2;
    const shapeSpecificLineWidthAdjustment = 1.5;

    const GLOBAL_LINE_WIDTH_ADJUSTMENT = 0.1;
    const verticalLineWidth = imageHeight * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredHeight;
    const horizontalLineWidth = imageWidth * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredWidth;
    const lineWidth = Math.min(verticalLineWidth, horizontalLineWidth);

    const magnification = Math.min((imageWidth - lineWidth)/requiredWidth, (imageHeight - lineWidth)/requiredHeight);
    const xOffset = lineWidth / 2;
    const yOffset = lineWidth / 2;

    // 转换坐标到图像坐标系（使用与绘制函数相同的转换）
    function xCoord(x) {
        return xOffset + x * magnification;
    }
    function yCoord(y) {
        return yOffset + y * magnification;
    }

    // 使用与绘制函数相同的中心点
    const cx = layers;
    const cy = layers;

    // 使用与绘制函数相同的 polarToXy 函数
    function polarToXy(angle, distance) {
        return [cx + distance * Math.sin(angle), cy - distance * Math.cos(angle)];
    }

    // 使用与绘制函数相同的 getCellCoords 函数
    function getCellCoords(l, c) {
        const cellsInLayer = cellCounts[l],
            anglePerCell = Math.PI * 2 / cellsInLayer,
            startAngle = anglePerCell * c,
            endAngle = startAngle + anglePerCell,
            innerDistance = l,
            outerDistance = l + 1;

        return [startAngle, endAngle, innerDistance, outerDistance];
    }

    // 获取扇形区域的参数
    const [startAngle, endAngle, innerDistance, outerDistance] = getCellCoords(layer, cellIndex);

    // 统一：所有层（包括中心层）使用相同的“扇形”逻辑
    // 对中心层等价于 inner=0, outer=1, 角度 0..2π

    const sectorStart = layer === 0 ? 0 : startAngle;
    const sectorEnd = layer === 0 ? Math.PI * 2 : endAngle;
    const ringInner = layer === 0 ? 0 : innerDistance;
    const ringOuter = layer === 0 ? 1 : outerDistance;

    // 计算边界框：考虑起止角 + 扇区内的基准角(0, π/2, π, 3π/2)
    function normalise(a) {
        let r = a % (Math.PI * 2);
        if (r < 0) r += Math.PI * 2;
        return r;
    }
    function angleSpan(start, end) {
        const s = normalise(start), e = normalise(end);
        return e >= s ? (e - s) : (Math.PI * 2 - s + e);
    }
    function angleInSector(a, start, end) {
        return isAngleInSector(normalise(a), normalise(start), normalise(end));
    }

    const candidateAngles = [sectorStart, sectorEnd, 0, Math.PI/2, Math.PI, 3*Math.PI/2];
    const uniqueAngles = [];
    candidateAngles.forEach(a => {
        const aa = normalise(a);
        if (angleInSector(aa, sectorStart, sectorEnd) && !uniqueAngles.some(x => Math.abs(x - aa) < 1e-6)) {
            uniqueAngles.push(aa);
        }
    });

    // 若为整圆或几乎整圆，直接用整圆边界
    const span = angleSpan(sectorStart, sectorEnd);
    const isFullCircle = (layer === 0) || (span >= 2 * Math.PI - 1e-6);
    let allPoints;
    if (isFullCircle) {
        allPoints = [
            [cx - ringOuter, cy],
            [cx + ringOuter, cy],
            [cx, cy - ringOuter],
            [cx, cy + ringOuter]
        ];
    } else {
        const innerPoints = uniqueAngles.map(a => polarToXy(a, ringInner));
        const outerPoints = uniqueAngles.map(a => polarToXy(a, ringOuter));
        allPoints = [...innerPoints, ...outerPoints];
        // 若内半径为0，加入中心点，避免边界退化
        if (ringInner === 0) {
            allPoints.push([cx, cy]);
        }
    }

    const xs = allPoints.map(p => xCoord(p[0]));
    const ys = allPoints.map(p => yCoord(p[1]));

    let minX = Math.floor(Math.min(...xs) - 0.5);
    let maxX = Math.ceil(Math.max(...xs) + 0.5);
    let minY = Math.floor(Math.min(...ys) - 0.5);
    let maxY = Math.ceil(Math.max(...ys) + 0.5);

    // 容差：半径与角度
    const epsR = Math.max(0, 0.5 / Math.max(1e-6, magnification)); // 半个像素对应的半径容差
    const epsA = 0.002; // 约 0.1°

    // 填充扇形区域内的像素
    for (let py = Math.max(0, minY); py <= Math.min(imageHeight - 1, maxY); py++) {
        for (let px = Math.max(0, minX); px <= Math.min(imageWidth - 1, maxX); px++) {
            const pixelCenterX = px + 0.5;
            const pixelCenterY = py + 0.5;

            // 将像素坐标转换回原始坐标系
            const origX = (pixelCenterX - xOffset) / magnification;
            const origY = (pixelCenterY - yOffset) / magnification;
            const dx = origX - cx;
            const dy = origY - cy;
            const pixelDistance = Math.sqrt(dx * dx + dy * dy);

            // 检查是否在正确的环形范围内（加入半径容差）
    if (pixelDistance + epsR >= ringInner && pixelDistance - epsR <= ringOuter) {
                // 若为整圆/近整圆：无需角度判断，直接填充
                let inSector = false;
                if (isFullCircle) {
                    inSector = true;
                } else {
                    // 计算角度 - 与 polarToXy 一致：x = cx + r*sin(a), y = cy - r*cos(a)
                    let pixelAngle = Math.atan2(dx, -dy);
                    if (pixelAngle < 0) pixelAngle += Math.PI * 2;
                    inSector = isAngleInSector(pixelAngle, sectorStart - epsA, sectorEnd + epsA);
                }
                if (inSector) {
                    const maskIndex = py * imageWidth + px;
                    maskData[maskIndex] = 1;
                }
            }
        }
    }
}

/**
 * 使用原始代码的逻辑计算每层的格子数量
 */
function cellCountsForLayers(layers) {
    const counts = [1];
    const rowRadius = 1 / layers;
    while (counts.length < layers) {
        const layer = counts.length;
        const previousCount = counts[layer - 1];
        const circumference = Math.PI * 2 * layer * rowRadius / previousCount;
        counts.push(previousCount * Math.round(circumference / rowRadius));
    }
    return counts;
}

/**
 * 检查角度是否在扇区范围内
 */
function isAngleInSector(angle, startAngle, endAngle) {
    // 标准化角度到 [0, 2π)
    while (angle < 0) angle += Math.PI * 2;
    while (angle >= Math.PI * 2) angle -= Math.PI * 2;
    while (startAngle < 0) startAngle += Math.PI * 2;
    while (startAngle >= Math.PI * 2) startAngle -= Math.PI * 2;
    while (endAngle < 0) endAngle += Math.PI * 2;
    while (endAngle >= Math.PI * 2) endAngle -= Math.PI * 2;

    // 处理正常情况和跨越0度的情况
    if (startAngle <= endAngle) {
        // 正常情况：角度范围不跨越0度
        return angle >= startAngle && angle <= endAngle;
    } else {
        // 跨越0度的情况：例如从350°到10°
        return angle >= startAngle || angle <= endAngle;
    }
}

/**
 * 判断点是否在三角形内（重心坐标法）
 */
function isPointInTriangle(px, py, x1, y1, x2, y2, x3, y3) {
    const denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    if (Math.abs(denom) < 1e-10) return false;

    const a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom;
    const b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom;
    const c = 1 - a - b;

    return a >= 0 && b >= 0 && c >= 0;
}

/**
 * 更精确的三角形内点判断（使用像素中心点和更高精度）
 */
function isPointInTriangleBarycentric(px, py, x1, y1, x2, y2, x3, y3) {
    // 使用标准的重心坐标公式 - 不取绝对值以保持顶点顺序信息
    const denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    if (Math.abs(denom) < 1e-10) return false; // 退化的三角形

    const a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom;
    const b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom;
    const c = 1 - a - b;

    // 使用小的容差来处理边界情况
    const eps = 1e-6;
    return a >= -eps && b >= -eps && c >= -eps;
}

/**
 * 判断点是否在多边形内（射线法）
 */
function isPointInPolygon(x, y, vertices) {
    let inside = false;
    for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i++) {
        const [xi, yi] = vertices[i];
        const [xj, yj] = vertices[j];

        if (((yi > y) !== (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

/**
 * 生成格子分割图：每个像素标记其所属格子的ID
 * @param {Object} maze - 迷宫对象
 * @param {number} imageWidth - 输出图像宽度
 * @param {number} imageHeight - 输出图像高度
 * @returns {Uint32Array} 格子ID数组，每个像素值为其所属格子的ID
 */
export function generateCellSegmentationMap(maze, imageWidth = 512, imageHeight = 512) {
    const segmentationMap = new Uint32Array(imageWidth * imageHeight);

    // 确定网格形状（使用与generateShapeAwareMask相同的逻辑）
    let shape;
    if (maze.metadata.cellShape) {
        shape = maze.metadata.cellShape;
    } else if (maze.metadata.grid && maze.metadata.grid.cellShape) {
        shape = maze.metadata.grid.cellShape;
    } else {
        if (maze.isSquare === true) {
            shape = SHAPE_SQUARE;
        } else if (maze.isSquare === false) {
            if (maze.metadata.layers !== undefined) {
                shape = SHAPE_CIRCLE;
            } else {
                let testCell = maze.getCellByCoordinates(1, 1);
                if (!testCell) {
                    testCell = maze.getCellByCoordinates(0, 0);
                }
                if (testCell) {
                    const neighborCount = testCell.neighbours.toArray().length;
                    if (neighborCount <= 3) {
                        shape = SHAPE_TRIANGLE;
                    } else {
                        shape = SHAPE_HEXAGON;
                    }
                } else {
                    shape = SHAPE_TRIANGLE;
                }
            }
        } else if (maze.metadata.layers !== undefined) {
            shape = SHAPE_CIRCLE;
        } else {
            shape = SHAPE_SQUARE;
        }
    }

    // 遍历每个格子，填充其对应的像素区域
    maze.forEachCell(cell => {
        if (cell && cell.metadata[METADATA_RAW_COORDS]) {
            const coords = cell.coords;
            const [centerX, centerY] = cell.metadata[METADATA_RAW_COORDS];

            // 为格子分配唯一ID
            const cellId = coordsToCellId(coords, maze, shape);

            // 填充格子区域为其ID
            fillCellSegmentation(segmentationMap, imageWidth, imageHeight, centerX, centerY, coords, shape, maze, cellId);
        }
    });

    return segmentationMap;
}

/**
 * 将格子坐标转换为唯一ID
 * 所有有效格子ID从1开始，0保留给背景
 */
function coordsToCellId(coords, maze, shape) {
    if (shape === SHAPE_CIRCLE) {
        // 圆形迷宫：[layer, cellIndex]
        const [layer, cellIndex] = coords;
        // 使用层和索引编码，+1确保从1开始
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
 * 根据形状填充格子分割图
 */
function fillCellSegmentation(segmentationMap, imageWidth, imageHeight, centerX, centerY, coords, shape, maze, cellId) {
    switch (shape) {
        case SHAPE_SQUARE:
            fillSquareSegmentation(segmentationMap, imageWidth, imageHeight, coords, maze, cellId);
            break;
        case SHAPE_TRIANGLE:
            fillTriangleSegmentation(segmentationMap, imageWidth, imageHeight, coords, maze, cellId);
            break;
        case SHAPE_HEXAGON:
            fillHexagonSegmentation(segmentationMap, imageWidth, imageHeight, coords, maze, cellId);
            break;
        case SHAPE_CIRCLE:
            fillCircleSegmentation(segmentationMap, imageWidth, imageHeight, coords, maze, cellId);
            break;
        default:
            fillSquareSegmentation(segmentationMap, imageWidth, imageHeight, coords, maze, cellId);
    }
}

/**
 * 填充方形格子分割图
 */
function fillSquareSegmentation(segmentationMap, imageWidth, imageHeight, coords, maze, cellId) {
    const [x, y] = coords;
    const gridWidth = maze.metadata.width;
    const gridHeight = maze.metadata.height;

    const cellPixelWidth = Math.max(1, Math.floor(imageWidth / gridWidth));
    const cellPixelHeight = Math.max(1, Math.floor(imageHeight / gridHeight));

    const halfWidth = Math.floor(cellPixelWidth / 2);
    const halfHeight = Math.floor(cellPixelHeight / 2);

    const centerX = (x + 0.5) * cellPixelWidth;
    const centerY = (y + 0.5) * cellPixelHeight;

    const startX = Math.max(0, Math.floor(centerX) - halfWidth);
    const endX = Math.min(imageWidth - 1, Math.floor(centerX) + halfWidth);
    const startY = Math.max(0, Math.floor(centerY) - halfHeight);
    const endY = Math.min(imageHeight - 1, Math.floor(centerY) + halfHeight);

    for (let py = startY; py <= endY; py++) {
        for (let px = startX; px <= endX; px++) {
            const index = py * imageWidth + px;
            segmentationMap[index] = cellId;
        }
    }
}

/**
 * 填充三角形格子分割图（复用fillTriangleMask的逻辑）
 */
function fillTriangleSegmentation(segmentationMap, imageWidth, imageHeight, coords, maze, cellId) {
    const [x, y] = coords;
    const verticalAltitude = Math.sin(Math.PI/3);

    function hasBaseOnSouthSide(x, y) {
        return (x + y) % 2;
    }

    function getCornerCoords(x, y) {
        let p1x, p1y, p2x, p2y, p3x, p3y;

        if (hasBaseOnSouthSide(x, y)) {
            p1x = x/2;
            p1y = (y+1) * verticalAltitude;
            p2x = (x+1)/2;
            p2y = p1y - verticalAltitude;
            p3x = p1x + 1;
            p3y = p1y;
        } else {
            p1x = x/2;
            p1y = y * verticalAltitude;
            p2x = (x+1)/2;
            p2y = p1y + verticalAltitude;
            p3x = p1x + 1;
            p3y = p1y;
        }

        return [p1x, p1y, p2x, p2y, p3x, p3y];
    }

    const [p1x, p1y, p2x, p2y, p3x, p3y] = getCornerCoords(x, y);

    const requiredWidth = 0.5 + maze.metadata.width/2;
    const requiredHeight = maze.metadata.height * verticalAltitude;
    const shapeSpecificLineWidthAdjustment = 0.8;

    const GLOBAL_LINE_WIDTH_ADJUSTMENT = 0.1;
    const verticalLineWidth = imageHeight * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredHeight;
    const horizontalLineWidth = imageWidth * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredWidth;
    const lineWidth = Math.min(verticalLineWidth, horizontalLineWidth);

    const magnification = Math.min((imageWidth - lineWidth)/requiredWidth, (imageHeight - lineWidth)/requiredHeight);
    const xOffset = lineWidth / 2;
    const yOffset = lineWidth / 2;

    function xCoord(x) {
        return xOffset + x * magnification;
    }
    function yCoord(y) {
        return yOffset + y * magnification;
    }

    const tp1x = xCoord(p1x);
    const tp1y = yCoord(p1y);
    const tp2x = xCoord(p2x);
    const tp2y = yCoord(p2y);
    const tp3x = xCoord(p3x);
    const tp3y = yCoord(p3y);

    const minX = Math.floor(Math.min(tp1x, tp2x, tp3x) - 0.5);
    const maxX = Math.ceil(Math.max(tp1x, tp2x, tp3x) + 0.5);
    const minY = Math.floor(Math.min(tp1y, tp2y, tp3y) - 0.5);
    const maxY = Math.ceil(Math.max(tp1y, tp2y, tp3y) + 0.5);

    for (let py = Math.max(0, minY); py <= Math.min(imageHeight - 1, maxY); py++) {
        for (let px = Math.max(0, minX); px <= Math.min(imageWidth - 1, maxX); px++) {
            const pixelCenterX = px + 0.5;
            const pixelCenterY = py + 0.5;

            if (isPointInTriangleBarycentric(pixelCenterX, pixelCenterY, tp1x, tp1y, tp2x, tp2y, tp3x, tp3y)) {
                const index = py * imageWidth + px;
                segmentationMap[index] = cellId;
            }
        }
    }
}

/**
 * 填充六边形格子分割图
 */
function fillHexagonSegmentation(segmentationMap, imageWidth, imageHeight, coords, maze, cellId) {
    const [x, y] = coords;
    const yOffset1 = Math.cos(Math.PI / 3);
    const yOffset2 = 2 - yOffset1;
    const yOffset3 = 2;
    const xOffset = Math.sin(Math.PI / 3);

    const requiredWidth = maze.metadata.width * 2 * xOffset + Math.min(1, maze.metadata.height - 1) * xOffset;
    const requiredHeight = maze.metadata.height * yOffset2 + yOffset1;
    const shapeSpecificLineWidthAdjustment = 1.5;

    const GLOBAL_LINE_WIDTH_ADJUSTMENT = 0.1;
    const verticalLineWidth = imageHeight * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredHeight;
    const horizontalLineWidth = imageWidth * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredWidth;
    const lineWidth = Math.min(verticalLineWidth, horizontalLineWidth);

    const magnification = Math.min((imageWidth - lineWidth)/requiredWidth, (imageHeight - lineWidth)/requiredHeight);
    const xOffsetDrawing = lineWidth / 2;
    const yOffsetDrawing = lineWidth / 2;

    function xCoord(x) {
        return xOffsetDrawing + x * magnification;
    }
    function yCoord(y) {
        return yOffsetDrawing + y * magnification;
    }

    function getCornerCoords(x, y) {
        const rowXOffset = Math.abs(y % 2) * xOffset,
            p1x = rowXOffset + x * xOffset * 2,
            p1y = yOffset1 + y * yOffset2,
            p2x = p1x,
            p2y = (y + 1) * yOffset2,
            p3x = rowXOffset + (2 * x + 1) * xOffset,
            p3y = y * yOffset2 + yOffset3,
            p4x = p2x + 2 * xOffset,
            p4y = p2y,
            p5x = p4x,
            p5y = p1y,
            p6x = p3x,
            p6y = y * yOffset2;

        return [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y];
    }

    const [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y] = getCornerCoords(x, y);

    const vertices = [
        [xCoord(p1x), yCoord(p1y)],
        [xCoord(p2x), yCoord(p2y)],
        [xCoord(p3x), yCoord(p3y)],
        [xCoord(p4x), yCoord(p4y)],
        [xCoord(p5x), yCoord(p5y)],
        [xCoord(p6x), yCoord(p6y)]
    ];

    const xs = vertices.map(v => v[0]);
    const ys = vertices.map(v => v[1]);
    const minX = Math.floor(Math.min(...xs) - 0.5);
    const maxX = Math.ceil(Math.max(...xs) + 0.5);
    const minY = Math.floor(Math.min(...ys) - 0.5);
    const maxY = Math.ceil(Math.max(...ys) + 0.5);

    for (let py = Math.max(0, minY); py <= Math.min(imageHeight - 1, maxY); py++) {
        for (let px = Math.max(0, minX); px <= Math.min(imageWidth - 1, maxX); px++) {
            if (isPointInPolygon(px + 0.5, py + 0.5, vertices)) {
                const index = py * imageWidth + px;
                segmentationMap[index] = cellId;
            }
        }
    }
}

/**
 * 填充圆形格子分割图
 */
function fillCircleSegmentation(segmentationMap, imageWidth, imageHeight, coords, maze, cellId) {
    const [layer, cellIndex] = coords;
    const layers = maze.metadata.layers;

    const cellCounts = cellCountsForLayers(layers);

    const requiredWidth = layers * 2;
    const requiredHeight = layers * 2;
    const shapeSpecificLineWidthAdjustment = 1.5;

    const GLOBAL_LINE_WIDTH_ADJUSTMENT = 0.1;
    const verticalLineWidth = imageHeight * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredHeight;
    const horizontalLineWidth = imageWidth * GLOBAL_LINE_WIDTH_ADJUSTMENT * shapeSpecificLineWidthAdjustment / requiredWidth;
    const lineWidth = Math.min(verticalLineWidth, horizontalLineWidth);

    const magnification = Math.min((imageWidth - lineWidth)/requiredWidth, (imageHeight - lineWidth)/requiredHeight);
    const xOffset = lineWidth / 2;
    const yOffset = lineWidth / 2;

    function xCoord(x) {
        return xOffset + x * magnification;
    }
    function yCoord(y) {
        return yOffset + y * magnification;
    }

    const cx = layers;
    const cy = layers;

    function polarToXy(angle, distance) {
        return [cx + distance * Math.sin(angle), cy - distance * Math.cos(angle)];
    }

    function getCellCoords(l, c) {
        const cellsInLayer = cellCounts[l],
            anglePerCell = Math.PI * 2 / cellsInLayer,
            startAngle = anglePerCell * c,
            endAngle = startAngle + anglePerCell,
            innerDistance = l,
            outerDistance = l + 1;

        return [startAngle, endAngle, innerDistance, outerDistance];
    }

    const [startAngle, endAngle, innerDistance, outerDistance] = getCellCoords(layer, cellIndex);

    const sectorStart = layer === 0 ? 0 : startAngle;
    const sectorEnd = layer === 0 ? Math.PI * 2 : endAngle;
    const ringInner = layer === 0 ? 0 : innerDistance;
    const ringOuter = layer === 0 ? 1 : outerDistance;

    function normalise(a) {
        let r = a % (Math.PI * 2);
        if (r < 0) r += Math.PI * 2;
        return r;
    }
    function angleSpan(start, end) {
        const s = normalise(start), e = normalise(end);
        return e >= s ? (e - s) : (Math.PI * 2 - s + e);
    }
    function angleInSector(a, start, end) {
        return isAngleInSector(normalise(a), normalise(start), normalise(end));
    }

    const candidateAngles = [sectorStart, sectorEnd, 0, Math.PI/2, Math.PI, 3*Math.PI/2];
    const uniqueAngles = [];
    candidateAngles.forEach(a => {
        const aa = normalise(a);
        if (angleInSector(aa, sectorStart, sectorEnd) && !uniqueAngles.some(x => Math.abs(x - aa) < 1e-6)) {
            uniqueAngles.push(aa);
        }
    });

    const span = angleSpan(sectorStart, sectorEnd);
    const isFullCircle = (layer === 0) || (span >= 2 * Math.PI - 1e-6);
    let allPoints;
    if (isFullCircle) {
        allPoints = [
            [cx - ringOuter, cy],
            [cx + ringOuter, cy],
            [cx, cy - ringOuter],
            [cx, cy + ringOuter]
        ];
    } else {
        const innerPoints = uniqueAngles.map(a => polarToXy(a, ringInner));
        const outerPoints = uniqueAngles.map(a => polarToXy(a, ringOuter));
        allPoints = [...innerPoints, ...outerPoints];
        if (ringInner === 0) {
            allPoints.push([cx, cy]);
        }
    }

    const xs = allPoints.map(p => xCoord(p[0]));
    const ys = allPoints.map(p => yCoord(p[1]));

    let minX = Math.floor(Math.min(...xs) - 0.5);
    let maxX = Math.ceil(Math.max(...xs) + 0.5);
    let minY = Math.floor(Math.min(...ys) - 0.5);
    let maxY = Math.ceil(Math.max(...ys) + 0.5);

    const epsR = Math.max(0, 0.5 / Math.max(1e-6, magnification));
    const epsA = 0.002;

    for (let py = Math.max(0, minY); py <= Math.min(imageHeight - 1, maxY); py++) {
        for (let px = Math.max(0, minX); px <= Math.min(imageWidth - 1, maxX); px++) {
            const pixelCenterX = px + 0.5;
            const pixelCenterY = py + 0.5;

            const origX = (pixelCenterX - xOffset) / magnification;
            const origY = (pixelCenterY - yOffset) / magnification;
            const dx = origX - cx;
            const dy = origY - cy;
            const pixelDistance = Math.sqrt(dx * dx + dy * dy);

            if (pixelDistance + epsR >= ringInner && pixelDistance - epsR <= ringOuter) {
                let inSector = false;
                if (isFullCircle) {
                    inSector = true;
                } else {
                    let pixelAngle = Math.atan2(dx, -dy);
                    if (pixelAngle < 0) pixelAngle += Math.PI * 2;
                    inSector = isAngleInSector(pixelAngle, sectorStart - epsA, sectorEnd + epsA);
                }
                if (inSector) {
                    const maskIndex = py * imageWidth + px;
                    segmentationMap[maskIndex] = cellId;
                }
            }
        }
    }
}