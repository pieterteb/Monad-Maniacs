#include <bits/stdc++.h>

using namespace std;

__int128 strToi128(string s) {
    __int128 res = 0;
    bool neg = false;
    size_t i = 0;
    if (s[i] == '-') {
        neg = true;
        ++i;
    }
    size_t slen = s.size();
    for (; i < slen; ++i)
        res = res * 10 + (s[i] - '0');
    return neg ? -res : res;
}
