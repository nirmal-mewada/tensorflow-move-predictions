import os
from datetime import time

import packages.tools as tools
import tensorflow_chessbot
from packages import sunfish
import chess
import chess.engine
import pyscreenshot as screen


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
    pos = tools.parseFEN(fen)
    print(' '.join(pos.board))
    searcher = sunfish.Searcher()

    # position = chess.Board()
    # epd_info = position.set_epd(fen)
    # print("Searching...")
    # move, score = searcher.search(pos, secs=args.time )
    # move = move if args.active == 'w' else (119 - move[0], 119 - move[1])
    # suggestedMove = sunfish.render(move[0]) + sunfish.render(move[1])
    # return suggestedMove, score
    engine = chess.engine.SimpleEngine.popen_uci("/Users/nirmal.s/Documents/Bundles_/stockfish-11-mac/Mac/stockfish-11-64")
    board = chess.Board(fen)
    limit = chess.engine.Limit(time=args.time)
    result = engine.play(board, limit)

    print(result.move)

    return limit, limit
    engine.quit()


screenshot_png = 'screenshot.png'
args = Object()
# args.filepath = "/Users/nirmal.s/Desktop/chess-input-2.png"
# args.filepath = "/Users/nirmal.s/Documents/Projects/ML/chess-learn/samples/stellung5.png"
args.filepath = screenshot_png
args.active = 'w'
args.unflip = 'False'
args.time = 3
args.predictor = tensorflow_chessbot.ChessboardPredictor()

while True:
    im = screen.grab()
    im.save(screenshot_png)
    score, suggestedMove = suggest_move(args)
    #print("Suggested move (score):", suggestedMove, score)
    input("Press Enter to continue...")
