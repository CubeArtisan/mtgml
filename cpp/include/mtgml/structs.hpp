#ifndef MTGML_STRUCTS_HPP
#define MTGML_STRUCTS_HPP
#include <array>
#include <cstdint>

constexpr std::size_t MAX_CUBE_SIZE = 1080;
constexpr std::size_t MAX_DECK_SIZE = 80;
constexpr std::size_t MAX_SIDEBOARD_SIZE = 48;
constexpr std::size_t MAX_CARDS_IN_PACK = 16;
constexpr std::size_t MAX_BASICS = 16;
constexpr std::size_t MAX_PICKED = 48;
constexpr std::size_t MAX_SEEN_PACKS = 48;
constexpr std::size_t READ_SIZE = 64;

using CubeCards = std::array<std::uint16_t, MAX_CUBE_SIZE>;
using Pack = std::array<std::int16_t, MAX_CARDS_IN_PACK>;
using CoordPair = std::array<std::int8_t, 2>;
using Coords = std::array<CoordPair, 4>;
using CoordWeights = std::array<float, 4>;
using Basics = std::array<std::int16_t, MAX_BASICS>;
using Pool = std::array<std::int16_t, MAX_PICKED>;
using SeenPacks = std::array<Pack, MAX_SEEN_PACKS>;
using SeenPackCoords = std::array<Coords, MAX_SEEN_PACKS>;
using SeenPackCoordWeights = std::array<CoordWeights, MAX_SEEN_PACKS>;
using MainDeck = std::array<std::uint16_t, MAX_DECK_SIZE>;
using Sideboard = std::array<std::uint16_t, MAX_SIDEBOARD_SIZE>;

struct Deck {
    std::array<std::uint16_t, MAX_DECK_SIZE> main;
    std::array<std::uint16_t, MAX_SIDEBOARD_SIZE> side;
};

struct Pick {
    Pack cards_in_pack{0};
    Basics basics{0};
    Pool pool{0};
    SeenPacks seen_packs{0};
    SeenPackCoords seen_coords{0};
    SeenPackCoordWeights seen_coord_weights{0};
    Coords coords{0};
    CoordWeights coord_weights{0};
    std::array<float, MAX_CARDS_IN_PACK> riskiness;
    std::int8_t is_trashed{0};
};

struct Draft {
    Basics basics{0};
    Pool pool{0};
    SeenPacks seen_packs{0};
    SeenPackCoords seen_coords{0};
    SeenPackCoordWeights seen_coord_weights{0};
    std::array<std::array<float, MAX_CARDS_IN_PACK>, MAX_SEEN_PACKS> riskiness;
    std::array<std::int8_t, MAX_SEEN_PACKS> is_trashed{0};
};
#endif
