#ifndef MTGML_STRUCTS_HPP
#define MTGML_STRUCTS_HPP
#include <array>
#include <cstdint>

constexpr std::size_t MAX_CUBE_SIZE = 1080;
constexpr std::size_t MAX_DECK_SIZE = 80;
constexpr std::size_t MAX_SIDEBOARD_SIZE = 48;

struct Deck {
    std::array<std::uint16_t, MAX_DECK_SIZE> main;
    std::array<std::uint16_t, MAX_SIDEBOARD_SIZE> side;
};
using CubeCards = std::array<std::uint16_t, MAX_CUBE_SIZE>;
#endif
