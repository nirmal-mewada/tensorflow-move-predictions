import sys

import cairosvg
import cv2

import tensorflow_chessbot
import chess
import chess.engine
import chess.svg
import pyscreenshot as screen
import matplotlib.pyplot as plt
from PIL import Image


class Object(object):
    pass


def reverse_fen(data):
    data = data.split(' ')
    data[0] = data[0][::-1]
    data = " ".join(data)
    return data


def suggest_move(args):
    fen = tensorflow_chessbot.main(args)
    fen = fen if args.active == 'b' else reverse_fen(fen)

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    board = chess.Board(fen)
    limit = chess.engine.Limit(time=args.time)
    result = engine.play(board, limit)
    engine.quit()
    return result, board


def arrow(s):
    tail = chess.SQUARE_NAMES.index(s[:2])
    head = chess.SQUARE_NAMES.index(s[2:]) if len(s) > 2 else tail
    return chess.svg.Arrow(tail, head)


screenshot_png = 'screenshot.png'
args = Object()
# args.filepath = "/Users/nirmal.s/Desktop/chess-input-2.png"
# args.filepath = "/Users/nirmal.s/Documents/Projects/ML/chess-learn/samples/stellung5.png"
args.filepath = screenshot_png
args.active = 'w'
args.unflip = 'False'
args.time = 3
args.size = 512
args.engine = "/Users/nirmal.s/Documents/Bundles_/stockfish-11-mac/Mac/stockfish-11-64"
args.predictor = tensorflow_chessbot.ChessboardPredictor()

image = None
while True:
    try:
        im = screen.grab(bbox=(10, 10, 1800, 1800))
        im.save(screenshot_png)
        result, board = suggest_move(args)
        if (board.is_game_over()):
            print("Game Over..!!!")
        else:
            print("Suggested move:", result.move)
        move = str(result.move)[:4]
        arrows = [arrow(s.strip()) for s in str(move).split(",") if s.strip()]
        flipped = False if args.active == 'w' else True
        svg_data = chess.svg.board(board, coordinates=False, flipped=flipped, lastmove=None, check=None,
                                   arrows=arrows,
                                   size=args.size, style=None)
        png_data = cairosvg.svg2png(bytestring=svg_data, write_to="tmp.png")

        image = cv2.imread("tmp.png")

        cv2.imshow("Input", image)
        cv2.waitKey(1)
    except Exception as e:
        print("Error ...." + str(e))
        pass
    # input("Press Enter to continue...")
