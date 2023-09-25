import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_xG_rolling(team, ax, color_for , color_ag ,X_ ,Y_for ,Y_ag):


    if Y_for.shape[0] == 0:
        raise ValueError(f"Team {team} is not present in the DataFrame")

    # ---- Compute rolling average
    # 
    # Y_for = Y_for.rolling(window = 5, min_periods = 0).mean() # min_periods is for partial avg.
    # Y_ag = Y_ag.rolling(window = 5, min_periods = 0).mean()

    # ---- Create auxiliary series for filling between curves

    X_aux = X_.copy()
    X_aux.index = X_aux.index * 10 # 9 aux points in between each match
    last_idx = X_aux.index[-1] + 1
    X_aux = X_aux.reindex(range(last_idx))
    X_aux = X_aux.interpolate()

    # --- Aux series for the xG created (Y_for)
    Y_for_aux = Y_for.copy()
    Y_for_aux.index = Y_for_aux.index * 10
    last_idx = Y_for_aux.index[-1] + 1
    Y_for_aux = Y_for_aux.reindex(range(last_idx))
    Y_for_aux = Y_for_aux.interpolate()

    # --- Aux series for the xG conceded (Y_ag)
    Y_ag_aux = Y_ag.copy()
    Y_ag_aux.index = Y_ag_aux.index * 10
    last_idx = Y_ag_aux.index[-1] + 1
    Y_ag_aux = Y_ag_aux.reindex(range(last_idx))
    Y_ag_aux = Y_ag_aux.interpolate()

    # --- Plotting our data

    # --- Remove spines and add gridlines

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(ls = "--", lw = 0.25, color = "#4E616C")

    # --- The data

    for_ = ax.plot(X_, Y_for, marker = "o", mfc = "white", ms = 4, color = color_for,label='全体残差')
    ag_ = ax.plot(X_, Y_ag, marker = "o", mfc = "white", ms = 4, color = color_ag,label='分组残差')

    # --- Fill between

    for index in range(len(X_aux) - 1):
        # Choose color based on which line's on top
        if Y_for_aux.iloc[index + 1] > Y_ag_aux.iloc[index + 1]:
            color = for_[0].get_color()
        else:
            color = ag_[0].get_color()

        # Fill between the current point and the next point in pur extended series.
        ax.fill_between([X_aux[index], X_aux[index+1]], 
                        [Y_for_aux.iloc[index], Y_for_aux.iloc[index+1]], 
                        [Y_ag_aux.iloc[index], Y_ag_aux.iloc[index+1]], 
                        color=color, zorder = 2, alpha = 0.2, ec = None)


    # --- Ensure minimum value of Y-axis is zero
    ax.set_ylim(0)

    # --- Adjust tickers and spine to match the style of our grid

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2)) # ticker every 2 matchdays
    # xticks_ = ax.xaxis.set_ticklabels([x - 1 for x in range(0, len(X_) + 3, 2)])

    ax.xaxis.set_tick_params(length = 2, color = "#4E616C", labelcolor = "#4E616C", labelsize = 6)
    ax.yaxis.set_tick_params(length = 2, color = "#4E616C", labelcolor = "#4E616C", labelsize = 6)

    ax.spines["bottom"].set_edgecolor("#4E616C")

    # --- Legend and team name

    Y_for_last = Y_for.iloc[-1]
    Y_ag_last = Y_ag.iloc[-1]

    # -- Add the team's name
    team_ = ax.text(
        x = 0, y = ax.get_ylim()[1] + ax.get_ylim()[1]/20,
        s = f'{team}',
        color = "#4E616C",
        va = 'center',
        ha = 'left',
        size = 7
    )

    # # -- Add the xG created label
    # for_label_ = ax.text(
    #     x = X_.iloc[-1] + 0.75, y = Y_for_last,
    #     s = f'{Y_for_last:,.1f} xGF',
    #     color = color_for,
    #     va = 'center',
    #     ha = 'left',
    #     size = 6.5
    # )
    # 
    # # -- Add the xG conceded label
    # ag_label_ = ax.text(
    #     x = X_.iloc[-1] + 0.75, y = Y_ag_last,
    #     s = f'{Y_ag_last:,.1f} xGA',
    #     color = color_ag,
    #     va = 'center',
    #     ha = 'left',
    #     size = 6.5
    # )

data = pd.read_excel('残差对照.xlsx')
a=data['全体']
b=data['分组']

x=pd.Series(range(1,len(a)+1))
fig = plt.figure(figsize=(5, 2), dpi = 180)
ax = plt.subplot(111)
plot_xG_rolling("分组前后残差对照", ax, color_for = "#00A752", color_ag = "black",X_=x,Y_for=a,Y_ag=b)
# plt.yticks(range(-1, 7, 1))
plt.xticks(range(0, 101, 5))
# plt.xlim([0.5,100])
# plt.ylim([-0.5,6.5])
plt.xlabel("人员编号（sub xx)")
plt.ylabel("残差（毫升）")
plt.legend()
# plt.savefig('90天mRS预测结果只包含首次影像.png', dpi=1000)
plt.show()

# fig = plt.figure(figsize=(5, 2), dpi = 180)
# ax = plt.subplot(111)
# plot_xG_rolling("90天mRS预测结果（包含首次+随访影像）", ax, color_for = "#153aab", color_ag = "#fdcf41",X_=x,Y_for=b,Y_ag=ab)
# plt.yticks(range(-1, 7, 1))
# plt.xticks(range(0, 101, 5))
# plt.xlim([0.5,100])
# plt.ylim([-0.5,6.5])
# plt.xlabel("人员编号（sub xx)")
# plt.ylabel("mRS等级")
# plt.legend()
# # plt.savefig('90天mRS预测结果包含首次+随访影像.png', dpi=1000)
# plt.show()







