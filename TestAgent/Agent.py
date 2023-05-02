import torch
import numpy as np
import copy
import os

class Agent:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weight_file_path = os.path.join(dir_path, 'weight.npy')
        self.weights = np.load(weight_file_path)
        self.moves = []
        self.agent = Ai()
        self.prev_add_info = None
        self.check_left = False
        self.check_drop = 0
        self.rotate_left_act = False
        self.check_intra_drop = False
    
    def choose_action(self, observation):
        board, piece, next_piece, offsetx = convert_state(observation)
        add_info = get_add_info(observation)

        if self.prev_add_info is None or not torch.all(self.prev_add_info == add_info).item():
            if self.check_drop < 3:
                self.check_drop += 1
                return 0
            if not self.check_left:
                self.check_left = True
                return 6
            self.moves.extend(self.agent.choose(board, piece, next_piece, offsetx, self.weights))
            self.check_intra_drop = True
            self.prev_add_info = add_info
            
            self.check_left = False
            self.check_drop = 0
        if len(self.moves) > 0:
            action = self.moves[0]
            if action == 'UP':
                self.rotate_left_act = True
                self.moves.pop(0)
                return 4
            
            if self.rotate_left_act:
                state1 = torch.tensor(observation[:, :17])
                state1 = torch.squeeze(state1)
                offsetx1 = get_offset(state1)
                if offsetx1 > 3: return 6
                self.rotate_left_act = False
            if action == 'LEFT':
                self.moves.pop(0) 
                return 6
            if action == 'RIGHT':
                self.moves.pop(0) 
                return 5
        else:
            if self.rotate_left_act:
                state1 = torch.tensor(observation[:, :17])
                state1 = torch.squeeze(state1)
                offsetx1 = get_offset(state1)
                if offsetx1 > 3: return 6
                self.rotate_left_act = False
            if self.check_intra_drop:
                self.check_intra_drop = False
                return 2
        return 0


class Field:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.field = [[0]*self.width]*self.height

    def size(self):
        return self.width, self.height

    def updateField(self, field):
        self.field = field

    @staticmethod
    def check_collision(field, shape, offset):
        off_x, off_y = offset
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                try:
                    if cell and field[ cy + off_y ][ cx + off_x ]:
                        return True
                except IndexError:
                    return True
        return False

    def projectPieceDown(self, piece, offsetX, workingPieceIndex):
        if offsetX+len(piece[0]) > self.width or offsetX < 0:
            return None
        #result = copy.deepcopy(self)
        offsetY = self.height
        for y in range(0, self.height):
            if Field.check_collision(self.field, piece, (offsetX, y)):
                offsetY = y
                break
        for x in range(0, len(piece[0])):
            for y in range(0, len(piece)):
                value = piece[y][x]
                if value > 0:
                    self.field[offsetY-1+y][offsetX+x] = -workingPieceIndex
        return self

    def undo(self, workingPieceIndex):
        self.field = [[0 if el == -workingPieceIndex else el for el in row] for row in self.field]

    def heightForColumn(self, column):
        width, height = self.size()
        for i in range(0, height):
            if self.field[i][column] != 0:
                return height-i
        return 0

    def heights(self):
        result = []
        width, height = self.size()
        for i in range(0, width):
            result.append(self.heightForColumn(i))
        return result

    def numberOfHoleInColumn(self, column):
        result = 0
        maxHeight = self.heightForColumn(column)
        for height, line in enumerate(reversed(self.field)):
            if height > maxHeight: break
            if line[column] == 0 and height < maxHeight:
                result+=1
        return result

    def numberOfHoleInRow(self, line):
        result = 0
        for index, value in enumerate(self.field[self.height-1-line]):
            if value == 0 and self.heightForColumn(index) > line:
                result += 1
        return result

    ################################################
    #                   HEURISTICS                 #
    ################################################

    def heuristics(self):
        heights = self.heights()
        maxColumn = self.maxHeightColumns(heights)
        return heights + [self.aggregateHeight(heights)] + self.numberOfHoles(heights) + self.bumpinesses(heights) + [self.completLine(), self.maxPitDepth(heights), self.maxHeightColumns(heights), self.minHeightColumns(heights)]

    def aggregateHeight(self, heights):
        result = sum(heights)
        return result

    def completLine(self):
        result = 0
        width, height = self.size()
        for i in range (0, height) :
            if 0 not in self.field[i]:
                result+=1
        return result

    def bumpinesses(self, heights):
        result = []
        for i in range(0, len(heights)-1):
            result.append(abs(heights[i]-heights[i+1]))
        return result

    def numberOfHoles(self, heights):
        results = []
        width, height = self.size()
        for j in range(0, width) :
            result = 0
            for i in range (0, height) :
                if self.field[i][j] == 0 and height-i < heights[j]:
                    result+=1
            results.append(result)
        return results

    def maxHeightColumns(self, heights):
        return max(heights)

    def minHeightColumns(self, heights):
        return min(heights)

    def maximumHoleHeight(self, heights):
        if self.numberOfHole(heights) == 0:
            return 0
        else:
            maxHeight = 0
            for height, line in enumerate(reversed(self.field)):
                if sum(line) == 0: break
                if self.numberOfHoleInRow(height) > 0:
                    maxHeight = height
            return maxHeight

    def rowsWithHoles(self, maxColumn):
        result = 0
        for line in range(0, maxColumn):
            if self.numberOfHoleInRow(line) > 0:
                result += 1
        return result

    def maxPitDepth(self, heights):
        return max(heights)-min(heights)



    @staticmethod
    def __offsetPiece(piecePositions, offset):
        piece = copy.deepcopy(piecePositions)
        for pos in piece:
            pos[0] += offset[0]
            pos[1] += offset[1]

        return piece

    def __checkIfPieceFits(self, piecePositions):
        for x,y in piecePositions:
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.field[y][x] >= 1:
                    return False
            else:
                return False
        return True

    def fitPiece(self, piecePositions, offset=None):
        if offset:
            piece = self.__offsetPiece(piecePositions, offset)
        else:
            piece = piecePositions

        field = copy.deepcopy(self.field)
        if self.__checkIfPieceFits(piece):
            for x,y in piece:
                field[y][x] = 1

            return field
        else:
            return None

#xoay các khối theo chiều kim đồng hồ
def rotate_clockwise(shape):
    return [ [ shape[y][x]
            for y in range(len(shape)) ]
        for x in range(len(shape[0]) - 1, -1, -1) ]

class Ai:

    @staticmethod
    #Hàm này sử dụng thuật toán tìm kiếm theo chiều sâu để xác định vị trí và xoay các khối Tetris để đạt được điểm số cao nhất. 
    def best(field, workingPieces, workingPieceIndex, weights, level):
        bestRotation = None
        bestOffset = None
        bestScore = None
        #chọn khối 
        workingPieceIndex = copy.deepcopy(workingPieceIndex)
        workingPiece = workingPieces[workingPieceIndex]
        shapes_rotation = { 4 : 4, 8 : 2, 12 : 2, 16 : 4, 20 : 4, 24 : 2, 28 : 1 }
        flat_piece = [val for sublist in workingPiece for val in sublist]
        hashedPiece = sum(flat_piece)

        for rotation in range(0, shapes_rotation[hashedPiece]):
            for offset in range(0, field.width):
                result = field.projectPieceDown(workingPiece, offset, level)
                if not result is None:
                    score = None
                    if workingPieceIndex == len(workingPieces)-1 :
                        heuristics = field.heuristics()
                        score = sum([a*b for a,b in zip(heuristics, weights)])
                    else:
                        _, _, score = Ai.best(field, workingPieces, workingPieceIndex + 1, weights, 2)

                    if bestScore is None or score > bestScore  :
                        bestScore = score
                        bestOffset = offset
                        bestRotation = rotation
                field.undo(level)
            workingPiece = rotate_clockwise(workingPiece)

        return bestOffset, bestRotation, bestScore

    @staticmethod
    #Hàm này dùng để chọn vị trí và xoay các khối Tetris dựa trên kết quả của hàm best
    def choose(initialField, piece, next_piece, offsetX, weights):
        field = Field(len(initialField[0]), len(initialField))
        field.updateField(copy.deepcopy(initialField))

        offset, rotation, _ = Ai.best(field, [piece, next_piece], 0, weights, 1)
        moves = []

        offset = offset - offsetX
        for _ in range(0, rotation):
            moves.append("UP")
        for _ in range(0, abs(offset)):
            if offset > 0:
                moves.append("RIGHT")
            else:
                moves.append("LEFT")
        #moves.append('RETURN')
        return moves

def get_offset(state):
    offsetx = -1
    for i in state[:, :10]:
        for j in range(len(i)):
            if i[j].item() > 0.4 and i[j].item() < 0.9:
                if offsetx == -1: offsetx = j
                else: offsetx = min(offsetx, j)
    if offsetx == -1: return 3
    return offsetx
    
def get_block_type(state):
    state = torch.squeeze(state)
    add_info = state[:, 10:17]
    info_tensor = torch.argmax(add_info[1:7], dim=1)
    info_tensor += 1; info_tensor
    return info_tensor[-1].item(), info_tensor[0].item()

def convert_state(state):
    tetris_shapes = [
        [[1, 1, 1, 1]],

        [[1, 1],
        [1, 1]],
        
        [[1, 0, 0],
        [1, 1, 1]],

        [[0, 0, 1],
        [1, 1, 1]],
        
        [[1, 1, 0],
        [0, 1, 1]],

        [[0, 1, 1],
        [1, 1, 0]],

        [[0, 1, 0],
        [1, 1, 1]],
    ]
    state = torch.tensor(state[:, :17])
    state = torch.squeeze(state)
    offsetx = get_offset(state)
    board = state[:, :10].to(int).tolist()
    piece, next_piece = get_block_type(state)
    piece, next_piece = tetris_shapes[piece - 1], tetris_shapes[next_piece - 1]
    return board, piece, next_piece, offsetx

def get_add_info(state):
    state = torch.tensor(state[:, :17])
    state = torch.squeeze(state)
    add_info = state[:, 10:17]
    info_tensor = torch.argmax(add_info[1:7], dim=1)
    info_tensor += 1; info_tensor
    return info_tensor
