import sys
import time

import cairosvg
import numpy as np
from cv2 import cv2

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


def arrow(s):
    tail = chess.SQUARE_NAMES.index(s[:2])
    head = chess.SQUARE_NAMES.index(s[2:]) if len(s) > 2 else tail
    return chess.svg.Arrow(tail, head)


def suggest_move(args):
    start_time = time.time()
    fen = tensorflow_chessbot.main(args)
    fen = fen if args.active == 'b' else reverse_fen(fen)
    if fen == args.last_fen:
        return None, None, None, None
    engine = chess.engine.SimpleEngine.popen_uci(args.engine_name)
    board = chess.Board(fen)
    limit = chess.engine.Limit(time=6)
    result = engine.play(board, limit, ponder=False)
    engine.quit()
    # print(" Prediction took : %.2d " % (time.time() - start_time))
    return result, board, fen,  ("Time: %.2fs" % (time.time() - start_time))


screenshot_png = 'screenshot.png'
args = Object()
args.active = 'w'
args.unflip = 'False'
args.time = 3
args.size = 512
args.img = None
args.last_fen = None
args.last_result = None
args.last_board = None
args.engine_name = "/Users/nirmal.s/Documents/Bundles_/stockfish-11-mac/Mac/stockfish-11-64"
args.predictor = tensorflow_chessbot.ChessboardPredictor()

last_fen = None
last_result = None
while True:
    try:
        im = screen.grab(bbox=(10, 10, 1800, 1800))
        args.img = im
        result, board, fen, time_took = suggest_move(args)
        if fen is None:
            print("(Cached) Move: %s, Fen: %s" % (str(args.last_result.move), args.last_fen))
            continue
        else:
            args.last_fen = fen
            args.last_result = result
            print("%s Move: %s, Fen: %s" % (time_took, str(args.last_result.move), args.last_fen))

        if (board.is_game_over()):
            print("Game Over..!!!")
            break
        move = str(args.last_result.move)[:4]
        arrows = [arrow(s.strip()) for s in str(move).split(",") if s.strip()]
        flipped = False if args.active == 'w' else True
        svg_data = chess.svg.board(board, coordinates=False, flipped=flipped, lastmove=None,
                                   check=None, arrows=arrows,
                                   size=args.size, style=None)
        png_data = cairosvg.svg2png(bytestring=svg_data)
        nparr = np.frombuffer(png_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        cv2.imshow("Your next move is...", img_np)
        cv2.waitKey(500)
    except Exception as e:
        print("Error ...." + str(e))
        pass

cv2.destroyAllWindows()
