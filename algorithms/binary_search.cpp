#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;

int bin_srch(vi& a, int t) {
    int l = 0, r = a.size() - 1, m;
    while (l <= r) {
        m = l + (r - l) / 2;
        if (a[m] == t)
            return m;
        else if (a[m] < t)
            l = m + 1;
        else
            r = m - 1;
    }
    return -1;
}
