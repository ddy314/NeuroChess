import chess
from scripts.novel_elite_collect_cloud import game_group,group_split


def test_game_family_split_and_temporal_sampling_both_sides():
 b=chess.Board();moves=[]
 for u in ['e2e4','e7e5','g1f3','b8c6','f1b5','a7a6']:moves.append(chess.Move.from_uci(u))
 g=game_group(b,moves);assert group_split(g)==group_split(game_group(b,list(moves)))
 assert game_group(b,moves)!=game_group(b,moves[:-1])
 selected=[ply for ply in range(14,161) if ply%3==0]
 assert {ply%2 for ply in selected}=={0,1}
