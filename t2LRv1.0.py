
from discord.ext import commands
import discord
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('dark_background')
l = []


test_bot = commands.Bot(command_prefix='@')


@test_bot.command()
async def elp(ctx):
    await ctx.send('*-select a stock based on naming convention, check* https://in.finance.yahoo.com/')


@test_bot.command()
async def dis(ctx, arg):
    await ctx.send('*-searching, if no `stock found` msg follows, try again in some time pepegapls*')
    l.append(arg)
    df = web.DataReader(arg, data_source='yahoo',
                        start='2010-01-01', end='2021-03-18')
    plt.figure(figsize=(16, 8))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Timeline', fontsize=18)
    plt.ylabel('Close Price($)', fontsize=18)
    plt.savefig('graph.png')
    await ctx.send(file=discord.File('graph.png'))
    # show the graph

    await ctx.send('*`stock found` hypers*')


@ test_bot.command()
async def t2h(ctx):
    await ctx.send('*-in test2 , you enter a num, that is number of days \n in future for which we predict\n the price. Format is @t2 num*')


@ test_bot.command()
async def t2(ctx, arg):
    n = int(arg)

    # df
    df = web.DataReader(l[-1], data_source='yahoo',
                        start='2016-01-01', end='2021-03-14')
    # new data frame with only close column
    df = df.filter(['Close'])
    # A variable for predicting 'n' days out into the future
    forecast_out = n  # 'n=30' days
    # Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Close']].shift(-forecast_out)
    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    x = np.array(df.drop(['Prediction'], 1))

    # Remove the last '30' rows
    x = x[:-forecast_out]
    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '30' rows
    y = y[:-forecast_out]
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Testing Model: Score returns the coefficient of determination R^2 of the prediction.
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
    # Print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)
    await ctx.send(lr_prediction)
    plt.figure(figsize=(16, 8))
    plt.title('Future Prediction')
    plt.plot(lr_prediction)
    plt.xlabel('+Days', fontsize=18)
    plt.ylabel('Close Price($)', fontsize=18)
    plt.savefig('graph.png')
    await ctx.send(file=discord.File('graph.png'))

test_bot.run('token here')

# add graph-->with jesus's help
# Visualise the data
# plt.figure(figsize=(16, 8))
# plt.title('Pred vs Actual')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.plot(dday, temp2, 'pls')
# plt.plot(dday, pred_price, 'pepe')
# plt.legend(['pls', 'work'], loc='lower right')
# plt.savefig('graph1.png')
# await ctx.send(file=discord.File('graph1.png'))
