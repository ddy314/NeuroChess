#pragma once
#include "vendor/chess.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cstdint>
namespace nc_rules {
using namespace chess;
constexpr int Unknown = 2;
int rawEp(const std::string &fen) {
  std::istringstream in(fen);
  std::string a, b, c, d;
  in >> a >> b >> c >> d;
  return d == "-" ? -1 : Square(d).index();
}
int nextEp(const Board &b, Move m) {
  return b.at(m.from()).type() == PieceType::PAWN &&
                 std::abs(m.to().index() - m.from().index()) == 16
             ? (m.to().index() + m.from().index()) / 2
             : -1;
}
int action(Move m, Color side) {
  int flip = side == Color::WHITE ? 0 : 56, to = m.to().index();
  if (m.typeOf() == Move::CASTLING)
    to = (to > m.from().index() ? 6 : 2) + (m.from().index() / 8) * 8;
  return (((m.from().index() ^ flip) * 64) + (to ^ flip)) * 5 +
         (m.typeOf() == Move::PROMOTION ? int(m.promotionType()) : 0);
}
// Match python-chess's insufficient-material semantics, including promoted
// bishops.
bool insufficient(const Board &b) {
  if (b.pieces(PieceType::PAWN) || b.pieces(PieceType::ROOK) ||
      b.pieces(PieceType::QUEEN))
    return false;
  auto knights = b.pieces(PieceType::KNIGHT),
       bishops = b.pieces(PieceType::BISHOP);
  if (!bishops)
    return knights.count() <= 1;
  if (knights)
    return false;
  int color = -1;
  while (bishops) {
    int s = bishops.pop();
    int c = (s / 8 + s % 8) % 2;
    if (color >= 0 && c != color)
      return false;
    color = c;
  }
  return true;
}
int terminal(Board &b, const Movelist &moves, bool repeated) {
  if (moves.empty())
    return b.inCheck() ? -1 : 0;
  if (insufficient(b) || b.halfMoveClock() >= 100 || b.isRepetition(2))
    return 0;
  if (repeated || b.halfMoveClock() == 99)
    for (auto m : moves) {
      bool fifty = b.halfMoveClock() == 99 &&
                   b.at(m.from()).type() != PieceType::PAWN && !b.isCapture(m);
      b.makeMove(m);
      bool draw = b.isRepetition(2) || (fifty && movegen::anylegalmoves(b));
      b.unmakeMove(m);
      if (draw)
        return 0;
    }
  return Unknown;
}
void encode(const Board &b, int ep, uint8_t *tokens, float *meta) {
  int flip = b.sideToMove() == Color::WHITE ? 0 : 56;
  for (int s = 0; s < 64; ++s) {
    auto p = b.at(Square(s));
    tokens[s ^ flip] =
        p == Piece::NONE
            ? 0
            : int(p.type()) + 1 + (p.color() == b.sideToMove() ? 0 : 6);
  }
  using Side = Board::CastlingRights::Side;
  auto cr = b.castlingRights();
  auto side = b.sideToMove();
  meta[0] = cr.has(side, Side::KING_SIDE);
  meta[1] = cr.has(side, Side::QUEEN_SIDE);
  meta[2] = cr.has(~side, Side::KING_SIDE);
  meta[3] = cr.has(~side, Side::QUEEN_SIDE);
  meta[4] = ep < 0 ? -1.f : float(ep ^ flip) / 63;
  meta[5] = std::min(150u, b.halfMoveClock()) / 150.f;
  meta[6] = b.isRepetition(1);
  meta[7] = b.isRepetition(2);
  meta[8] = b.inCheck();
  meta[9] = side == Color::WHITE;
  meta[10] = std::min(300u, b.fullMoveNumber()) / 300.f;
  meta[11] = ep >= 0;
}
} // namespace nc_rules
