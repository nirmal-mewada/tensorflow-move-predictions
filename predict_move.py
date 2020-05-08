import tensorflow_chessbot
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


    engine = chess.engine.SimpleEngine.popen_uci("/Users/nirmal.s/Documents/Bundles_/stockfish-11-mac/Mac/stockfish-11-64")
    board = chess.Board(fen)
    print(board.is_game_over())
    limit = chess.engine.Limit(time=args.time)
    result = engine.play(board, limit)

    print(result.move)
    engine.quit()
    return result



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
    #im = screen.grab()
    #im.save(screenshot_png)
    result = suggest_move(args)
    print("Suggested move:", result.move)
    input("Press Enter to continue...")
