# hexo-ai

Attempt to do a vaguely alphago-ish thing with the HeXO game

## Game rules

Hex Tic-Tac-Toe, also known as HeXO, is a game played on an infinite hex grid.
The first player to get 6 stones in a row, along any of the three axes, wins.
On the first turn, X places a single stone.
On subsequent turns, players alternate placing two stones each.

Basically
```python
player = 'x'
place_stone(player)
while neither_player_has_6_in_a_row():
    player = 'o' if player == 'x' else 'x'
    place_stone(player)
    place_stone(player)
```

## Notable math stuff

This is not a maker-maker game due to the turn structure.
