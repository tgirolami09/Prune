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
<tr>
<td> 16 input buckets </td>
<td>

```
(20000 nodes, 1t, 16MB):
10.96 +/- 6.43
15.91 +/- 9.33
[133, 606, 1065, 680, 180]
[0.00, 5.00]
```

</td>

<td>

```
(10+0.1, 1t, 16MB):
7.80 +/- 4.92
13.16 +/- 8.30
[75, 778, 1537, 867, 106]
[0.00, 5.00]
```

</td>
<td> 

```
(60+0.6, 1t, 128MB):
15.21 +/- 6.88
29.02 +/- 13.11
[6, 279, 674, 371, 19]
[0.00, 5.00]
```

</td> <td> no need to be tested </td>
</tr>
</table>