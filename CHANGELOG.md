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
<td> change </td> <td> faster </td> <td> STC </td> <td> LTC </td> <td> slower </td>
</tr>
<tr>
<td> fractional depth </td>
<td> not need to be tested </td>
<td>

```
(10+0.1, 1t, 16MB):
2.74 +/- 2.20
4.87 +/- 3.92
[236, 3616, 7237, 3668, 329]
[0.00, 5.00]
```

</td>
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
<td> 

```
(10+0.1, 1t, 16MB):
-4.49 +/- 4.73
-7.75 +/- 8.16
[90, 877, 1615, 831, 68]
[0.00, 5.00]
```
</td> <td> not tested </td>
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
<tr>
<td> merged king planes (same bench, reduce binary size) </td>
<td>

```
(4+0.04, 1t, 16MB):
-0.57 +/- 1.85
-0.98 +/- 3.15
[643, 5516, 11092, 5421, 652]
[-5.00, 0.00]
```
</td>
<td> no need to be tested </td> <td> no need to be tested </td> <td> no need to be tested </td>
</tr>
<tr>
<td>complexity tm</td>
<td>no need to be tested</td>
<td>

```
(10+0.1, 1t, 16MB):
3.87 +/- 2.97
7.18 +/- 5.51
[100, 1694, 3911, 1790, 137]
[0.00, 5.00] (passed)
```
</td>
<td>

```
(60+0.6, 1t, 128MB):
3.44 +/- 2.65
7.09 +/- 5.45
[19, 1685, 4265, 1822, 28]
[0.00, 5.00] (passed)
```
</td><td> no need to be tested </td>
</tr>
<tr>
<td>index mainhist by threat</td> <td> no need to be tested </td>
<td>

```
(10+0.1, 1t, 16MB):
14.47 +/- 6.76
26.99 +/- 12.58
[16, 290, 742, 390, 27]
[0.00, 5.00] (pass)
```
</td>
<td>

```
(60+0.6, 1t, 128MB):
15.76 +/- 6.80
32.11 +/- 13.83
[1, 233, 640, 333, 6]
[0.00, 5.00] (pass)
```
</td><td> no need to be tested </td>
</tr>
</table>