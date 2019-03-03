import numpy as np
import pandas as pd
from ggplot import *

from BlackScholes import BsCall

k = 1000

x = np.empty([k, 5])

for i in range(0, k):
    x[i, :] = BsCall(100.0, 0.2, 1.0, 100.0, 0.03, 252, 100000)[0, :]

x = pd.DataFrame(data=x, columns=("Price", "Delta", "Rho", "Theta", "Vega"))

print(x)

TruePrice = BsCall(100.0, 0.2, 1.0, 100.0, 0.03, 252, 10)[1, 0]
TrueDelta = BsCall(100.0, 0.2, 1.0, 100.0, 0.03, 252, 10)[1, 1]
TrueRho = BsCall(100.0, 0.2, 1.0, 100.0, 0.03, 252, 10)[1, 2]
TrueTheta = BsCall(100.0, 0.2, 1.0, 100.0, 0.03, 252, 10)[1, 3]
TrueVega = BsCall(100.0, 0.2, 1.0, 100.0, 0.03, 252, 10)[1, 4]

p = ggplot(aes(x="Price"), data=x) + geom_histogram(bins=30) + ggtitle("Histogram of Monte Carlo prices") + geom_vline(
    x=TruePrice, linetype='dashed', color='red') + labs("Price", "Count")
t = theme_gray()
t._rcParams['font.size'] = 18
p = p + t
p.save("/Users/FrederikKryger/Library/Mobile Documents/com~apple~CloudDocs/Mat-Øk/Speciale/Tex/Figures/PriceHist.png")
print(p)
p = ggplot(aes(x="Delta"), data=x) + geom_histogram(bins=30) + ggtitle("Histogram of Monte Carlo AD Delta") + labs(
    "Delta", "Count") + geom_vline(x=TrueDelta, linetype='dashed', color='red')
t = theme_gray()
t._rcParams['font.size'] = 18
p = p + t
p.save("/Users/FrederikKryger/Library/Mobile Documents/com~apple~CloudDocs/Mat-Øk/Speciale/Tex/Figures/DeltaHist.png")

p = ggplot(aes(x="Rho"), data=x) + geom_histogram(bins=30) + ggtitle("Histogram of Monte Carlo AD Rho") + labs("Rho",
                                                                                                               "Count") + geom_vline(
    x=TrueRho, linetype='dashed', color='red')
t = theme_gray()
t._rcParams['font.size'] = 18
p = p + t
p.save("/Users/FrederikKryger/Library/Mobile Documents/com~apple~CloudDocs/Mat-Øk/Speciale/Tex/Figures/RhoHist.png")

p = ggplot(aes(x="Theta"), data=x) + geom_histogram(bins=30) + ggtitle("Histogram of Monte Carlo AD Theta") + labs(
    "Theta", "Count") + geom_vline(x=TrueTheta, linetype='dashed', color='red')
t = theme_gray()
t._rcParams['font.size'] = 18
p = p + t
p.save("/Users/FrederikKryger/Library/Mobile Documents/com~apple~CloudDocs/Mat-Øk/Speciale/Tex/Figures/ThetaHist.png")

p = ggplot(aes(x="Vega"), data=x) + geom_histogram(bins=30) + ggtitle("Histogram of Monte Carlo AD Vega") + labs("Vega",
                                                                                                                 "Count") + geom_vline(
    x=TrueVega, linetype='dashed', color='red')
t = theme_gray()
t._rcParams['font.size'] = 18
p = p + t
p.save("/Users/FrederikKryger/Library/Mobile Documents/com~apple~CloudDocs/Mat-Øk/Speciale/Tex/Figures/VegaHist.png")
