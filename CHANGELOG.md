# Changelog
all tests are using UHO_Lichess_4852_v1.epd book

format : 
```
(TC, threads, hash)
elo
nElo
pentanomial
nElo bounds
```

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
(60+0.6, 1t, 128MB):
4.98 +/- 3.49
9.89 +/- 6.94
[17, 1065, 2534, 1163, 37]
[0.00, 5.00]
```

</td>
<td> no need to be tested </td>
</tr>
<tr>
<td>640 L1 net</td>
<td>

```
(20000 nodes, 1t, 16MB):
18.59 +/- 8.67
26.92 +/- 12.52
[69, 317, 588, 395, 109]
[0.00, 5.00]
```

</td>
<td> not tested </td> <td> not tested </td>
<td>

```
(20+0.2, 8t, 1024MB):
15.43 +/- 6.84
30.10 +/- 13.32
[8, 252, 677, 356, 14]
[0.00, 5.00]
```

</td>
</tr>
</table>