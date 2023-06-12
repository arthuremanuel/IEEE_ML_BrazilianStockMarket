# IEEE_ML_BrazilianStockMarket
<h1>Resources for the manuscript "Using Machine Learning to Prevent Losses in the Brazilian Stock Market During the Covid-19 Pandemic" published in IEEE Latin America Transactions.
</h1>

<img src="05_PNG_Graphical Abstract.png"/>

The code is available in Python programming language, and each file is presented in the following.

1. Main.py: this is the main file of the project, in which we load the .csv data, run the proposed Machine Learning models and invoke the libraries in order to produce the results of the paper.
2. Trader.py: this file contains the code related to the Backtrader library usage, in which the investment simulation is performed.
3. Trader_Strategy_ML.py: this file contains the code related to the investment strategy based on the Machine Learning models.
4. ./Data/^BVSP - 2020.csv: this file contains the dataset used in the paper, considering the Ibovespa index.

<h2>Instructions</h2>
How to run the code:
First, install the Python programming language and all the dependencies needed for this project:
1. MatPlotLib
2. Numpy
3. Scikit-Learn
4. Backtrader
5. Ta-Lib.

Then, in the command line, type:
> python Main.py

