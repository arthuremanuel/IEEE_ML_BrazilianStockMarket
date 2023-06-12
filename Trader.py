from __future__ import (absolute_import, division, print_function,
                        unicode_literals)



import backtrader as bt
from datetime import datetime

import backtrader as bt
from _datetime import date
import datetime  

def saveplots(cerebro, numfigs=1, iplot=True, start=None, end=None,
             width=10, height=9, dpi=300, tight=True, use=None, file_path = '', **kwargs):

        from backtrader import plot
        if cerebro.p.oldsync:
            plotter = plot.Plot_OldSync(**kwargs)
        else:
            plotter = plot.Plot(**kwargs)

        figs = []
        for stratlist in cerebro.runstrats:
            for si, strat in enumerate(stratlist):
                rfig = plotter.plot(strat, figid=si * 100,
                                    numfigs=numfigs, iplot=iplot,
                                    start=start, end=end, use=use)
                
                figs.append(rfig)

        for fig in figs:
            for f in fig:
                f.set_figwidth(width)
                f.savefig('./Output/'+file_path.replace('\'','').replace(',','').replace(' ','').replace('{','').replace('}','').replace('00:00:00)','').replace('Timestamp(','').replace(':',''), bbox_inches='tight')
        return figs


def createRunCerebro(data, strategy, strategyName, plotName, df = 'empty'):
    print('##################### ',strategyName,' ##################')
    cerebro = bt.Cerebro()

    cerebro.addstrategy(strategy, d0 = df)
    cerebro.adddata(data)
    cerebro.broker.setcash(200000.0)
    cerebro.broker.setcommission(commission=0.00) # Set the commission
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sr", timeframe=bt.TimeFrame.Days)
    strategies = cerebro.run(maxcpus=1) # Run over everything
    firstStrat = strategies[0]
    if plotName != None:
        saveplots(cerebro, file_path = '_savefig_'+plotName+'_.png') #run it #cerebro.plot(savefig=True, figfilename='backtrader-plot.png')
        print('Saving plot...')
    # print the analyzers
    #print(firstStrat.analyzers)
    report = {#'Cash\t' + str(cerebro.broker.get_cash()) +
              'Value' : cerebro.broker.get_value(),
              'Maximum Drawdown': firstStrat.analyzers.dd.get_analysis()['max']['drawdown'],
              'Sharpe Ratio': firstStrat.analyzers.sr.get_analysis()['sharperatio']}
    
    return report

class BuyAndHold(bt.Strategy):
    params = (
        ('printlog', False),
        ('d0', None )
    )
    
    def start(self):
        self.val_start = self.broker.get_cash()  # keep the starting cash

    def nextstart(self):
        # Buy all the available cash
        size = int(self.broker.get_cash() / self.data)
        self.buy(size=size)

    def stop(self):
        print('\tEnding Value\t',self.broker.get_value())
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print('\tROI\t{:.2f}%'.format(100.0 * self.roi))

# Create a Strategy
class MovAVGStrategy(bt.Strategy):
    params = (
        ('ma_low', 17),
        ('ma_high', 34),
        ('printlog', False),
        ('d0', None )
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma_low = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_low)
        self.sma_high = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_high)

    def start(self):
        self.val_start = self.broker.get_cash()  # keep the starting cash
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])

        if self.order: # Check if an order is pending ... if yes, we cannot send a 2nd one
            return

        if not self.position:  # Check if we are in the market
            if  self.sma_low[0] > self.sma_high[0]:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy()
        else:
            if self.sma_low[0] < self.sma_high[0]:
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()

    def stop(self):
        self.log('(MA Period %2d, %2d) \nEnding Value %.2f' %
                 (self.params.ma_low, self.params.ma_high , self.broker.getvalue()), doprint=True)
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print('ROI:        {:.2f}%'.format(100.0 * self.roi))


if __name__ == '__main__':
    data = bt.feeds.YahooFinanceCSVData(
        dataname='./Data/^BVSP (2).csv', 
        fromdate=datetime.datetime(2020, 1, 1),
        # Do not pass values before this date
        todate=datetime.datetime(2021, 1, 1),)

    createRunCerebro(data, MovAVGStrategy, 'MA Crossover', None,False) #graphname   
    createRunCerebro(data, BuyAndHold, 'Buy Hold', None,False) #graphname