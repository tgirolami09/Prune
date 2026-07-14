# Changelog
## unreleased
<table>
<tr>
<td> change </td> <td> fixed nodes </td> <td> STC </td> <td> LTC </td> <td> more </td>
</tr>
<tr>
<td> fractional depth </td>
<td> not need to be tested </td> <td> not tested</td>
<td>

```
Results of prunefdepth_tuned2_try3 vs prunedev (60+0.6, 1t, 128MB, UHO_Lichess_4852_v1.epd):
Elo: 4.98 +/- 3.49, nElo: 9.89 +/- 6.94
LOS: 99.74 %, DrawRatio: 52.62 %, PairsRatio: 1.11
Games: 9632, Wins: 2162, Losses: 2024, Draws: 5446, Points: 4885.0 (50.72 %)
Ptnml(0-2): [17, 1065, 2534, 1163, 37], WL/DD Ratio: 0.57
LLR: 2.95 (100.2%) (-2.94, 2.94) [0.00, 5.00]
```

</td>
<td> no need to be tested </td>
</tr>
<tr>
<td>640 L1 net</td>
<td>

```
Results of prunephalaina vs prunedev (20000 nodes, 1t, 16MB, UHO_Lichess_4852_v1.epd):
Elo: 18.59 +/- 8.67, nElo: 26.92 +/- 12.52
LOS: 100.00 %, DrawRatio: 39.78 %, PairsRatio: 1.31
Games: 2956, Wins: 966, Losses: 808, Draws: 1182, Points: 1557.0 (52.67 %)
Ptnml(0-2): [69, 317, 588, 395, 109], WL/DD Ratio: 1.50
LLR: 2.97 (100.9%) (-2.94, 2.94) [0.00, 5.00]
```

</td>
<td> not tested </td> <td> not tested </td>
<td>

```
Results of prunephalaina vs prunedev (20+0.2, 8t, 1024MB, UHO_Lichess_4852_v1.epd):
Elo: 15.43 +/- 6.84, nElo: 30.10 +/- 13.32
LOS: 100.00 %, DrawRatio: 51.80 %, PairsRatio: 1.42
Games: 2614, Wins: 653, Losses: 537, Draws: 1424, Points: 1365.0 (52.22 %)
Ptnml(0-2): [8, 252, 677, 356, 14], WL/DD Ratio: 0.66
LLR: 2.96 (100.6%) (-2.94, 2.94) [0.00, 5.00]
```

</td>
</tr>
</table>