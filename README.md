# turkish-llm
playground to create turkish llms

# Leaderboard

| Models     | max-len  | ntok      | nchar      | nvoc   | sum_nll       | sum_bits     | nll/tok | nll/char | token_ppl | char_ppl | bits/tok | bits/char |
|------------|----------|-----------|------------|--------|---------------|--------------|---------|----------|-----------|----------|----------|-----------|
| llama2-7b  |    4096  | 6,672,520 | 13,488,318 | 32,000 |  14,809,664.0 | 21,365,828.0 | 2.2195  | 1.097962 | 9.2027    | 2.9980   | 3.2021   | 1.5840    |
| mistral-7b |    4096  | 6,679,977 | 13,488,318 | 32,000 |  11,988,665.0 | 17,295,988.0 | 1.7653  | 0.888818 | 5.8438    | 2.4322   | 2.5469   | 1.2822    |
| vicuna-7b  |    4096  | 6,672,520 | 13,488,318 | 32,000 |  14,571,667.0 | 21,022,472.0 | 2.1838  | 1.080317 | 8.8802    | 2.9456   | 3.1506   | 1.5585    |
| llama2-7b  |    2048  | 6,672,520 | 13,488,318 | 32,000 |  13,262,011.0 | 19,133,038.0 | 1.9875  | 0.983222 | 7.2976    | 2.6730   | 2.8674   | 1.4184    |
| mistral-7b |    2048  | 6,679,977 | 13,488,318 | 32,000 |  12,210,814.0 | 17,616,480.0 | 1.7980  | 0.905288 | 6.0381    | 2.4726   | 2.5941   | 1.3060    |
| vicuna-7b  |    2048  | 6,672,520 | 13,488,318 | 32,000 |  14,809,664.0 | 21,365,828.0 | 2.2195  | 1.097962 | 9.2027    | 2.9980   | 3.2020   | 1.5840    |
| falcon-7b  |    2048  | 6,486,017 | 13,488,318 | 65,024 |  15,143,924.0 | 21,848,064.0 | 2.3348  | 1.122743 | 10.327    | 3.0732   | 3.3684   | 1.6197    |
| phi-1.5    |    2048  | 6,730,586 | 13,488,318 | 51,200 |  26,347,170.0 | 38,010,932.0 | 3.9145  | 1.953332 | 50.126    | 7.0521   | 5.6474   | 2.8180    |
| pythia-6.9b|    2048  | 6,149,190 | 13,488,318 | 50,432 |  12,318,955.0 | 17,772,496.0 | 2.0033  | 0.913305 | 7.4138    | 2.4925   | 2.8902   | 1.3176    |
| opt-6.7b   |    2048  | 6,735,965 | 13,488,318 | 50,272 |  12,028,472.0 | 17,353,416.0 | 1.7857  | 0.891769 | 5.9638    | 2.4394   | 2.5762   | 1.2865    |
| mpt-7b     |    1024  | 6,149,190 | 13,488,318 | 50,432 |  13,496,818.0 | 19,471,792.0 | 2.1948  | 1.000630 | 8.9790    | 2.7199   | 3.1665   | 1.4436    |
| gpt2-xl    |    1024  | 6,730,965 | 13,488,318 | 50,257 |  23,702,484.0 | 34,195,456.0 | 3.5214  | 1.757260 | 33.832    | 5.7965   | 5.0803   | 2.5351    |


| Models     | max-len  | ntok      | nchar      | nvoc   | sum_nll       | sum_bits     | nll/tok | nll/char | token_ppl | char_ppl | bits/tok | bits/char |
|------------|----------|-----------|------------|--------|---------------|--------------|---------|----------|-----------|----------|----------|-----------|
| xglm-7.5b  |    2048  | 3,308,314 | 13,488,318 | 256,008|   8,505,928.0 | 12,271,460.0 | 2.5710  | 0.630614 | 13.079    | 1.8787   | 3.7092   | 0.9097    |
| xglm-4.5b  |    2048  | 3,308,314 | 13,488,318 | 256,008|   9,073,295.0 | 13,089,998.0 | 2.7425  | 0.672677 | 15.526    | 1.9594   | 3.9566   | 0.9704    |
| xglm-2.9b  |    2048  | 3,308,314 | 13,488,318 | 256,008|   8,970,696.0 | 12,941,978.0 | 2.7115  | 0.665071 | 15.052    | 1.9446   | 3.9119   | 0.9594    |
| xglm-1.7b  |    2048  | 3,308,314 | 13,488,318 | 256,008|   9,517,353.0 | 13,730,638.0 | 2.8767  | 0.705599 | 17.757    | 2.0250   | 4.1503   | 1.0179    |
| xglm-564M  |    2048  | 3,308,314 | 13,488,318 | 256,008|  10,704,995.0 | 15,444,043.0 | 3.2357  | 0.793649 | 25.426    | 2.2114   | 4.6682   | 1.1449    |
| mGPT       |    2048  | 4,393,160 | 13,488,318 | 100,000|  21,218,566.0 | 30,611,920.0 | 4.8299  | 1.573106 | 125.19    | 4.8216   | 6.9680   | 2.2695    |
| bloomz-7b1 |    2048  | 5,459,180 | 13,488,318 | 250,880|  20,515,212.0 | 29,597,194.0 | 3.7579  | 1.520961 | 42.859    | 4.5766   | 5.4215   | 2.1942    |
| blz-7b1-mt |    2048  | 5,459,180 | 13,488,318 | 250,880|  20,547,892.0 | 29,644,342.0 | 3.7639  | 1.523384 | 43.116    | 4.5877   | 5.4301   | 2.1977    |
