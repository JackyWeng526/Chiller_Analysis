# %%
# Test parameters
# Filter conditions
limitations = dict(
    # KW/RT
    KWRT_upper_limit = 2 ,
    KWRT_lower_limit = 0 ,
    # Chiller KW
    KW_upper_limit = 300 , # 800
    KW_lower_limit = 10 ,
    # Chiller RT
    RT_upper_limit = 550 , # 800
    RT_lower_limit = 0 ,
    # Temperature of chilled water supply
    TCHS_upper_limit = 10 ,
    TCHS_lower_limit = 0 ,
    # Temperature of cooling water supply
    TCWS_upper_limit = 45 ,
    TCWS_lower_limit = 0 ,
    # Delta of chilled water
    TCH_delta_upper_limit = 20 ,
    TCH_delta_lower_limit = 0 ,
)

# Parameters for regression and analysis
reg_pars = dict(
    # timestep
    resample_step = "3min"
)

# 繪圖設定
parameters = dict(
    # KW/RT axes, ticks, shifts
    KWRT_cal_upper_limit = 1.2 , 
    KWRT_cal_lower_limit = 0.2 , 
    KWRT_cal_ticks = 0.4 ,
    KWRT_cal_shift = 0.1 ,
    # Chiller KW axes, ticks, shifts
    KW_upper_limit = 400 ,
    KW_lower_limit = 50 ,
    KW_ticks = 50 ,
    KW_shift = 10 ,
    # Chiller RT axes, ticks, shifts
    RT_cal_upper_limit = 800 , 
    RT_cal_lower_limit = 0 ,
    RT_cal_ticks = 200 ,
    RT_cal_shift = 10 ,
    # Chiller TCHS axes, ticks, shifts
    TCHS_upper_limit = 10 ,
    TCHS_lower_limit = 6 ,
    TCHS_ticks = 1 ,
    TCHS_shift = 0.5 ,
    # Chiller TCWS axes, ticks, shifts
    TCWS_upper_limit = 40 ,
    TCWS_lower_limit = 20 ,
    TCWS_ticks = 2 ,
    TCWS_shift = 0.5 ,
)

# Field & data parameters
# For default test data
field_name = "HanMingTech"
weather_data = "CWB_Hsinchu_2020-2021.csv"
Chiller_name = "P1_CH02" # For test
Chiller_list = ["P1_CH01", "P1_CH02", "P2_CH01"] # For test
Chiller_data_name = "P1_CH02.csv" # For test


# %%
# Import packages
import pandas as pd
import numpy as np
import datetime
from pandas import Timestamp
import os

# Plot related
import plotly.graph_objects as go
import chart_studio.plotly as py
import cufflinks as cf
import seaborn as sns
import matplotlib.pyplot as plt
cf.go_offline()

# Analysis related
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import lightgbm as lgb

# Create data paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__)) # root path of this pycode
DATA_PATH = os.path.join(BASE_PATH, "..", "data") # data files' path

# List the chillers' data
chiller_data_file = [f for f in os.listdir(DATA_PATH) if ("CH" in f) & (f.endswith(".csv"))]
print(F"chiller data files: {chiller_data_file}")


# %%
# Read the chiller data
def read_Chiller_data(Chiller_name):
    Chiller_data = pd.read_csv(os.path.join(DATA_PATH, F"{Chiller_name}.csv"), index_col="DateTime").astype(float)
    Chiller_data.index = pd.to_datetime(Chiller_data.index)

    # Filter the data with available values for analysis
    # Related to chilled water
    TCH_delta = Chiller_data.loc[:, "TCHR"] - Chiller_data.loc[:, "TCHS"]
    mask_TCHS = (Chiller_data.loc[:, "TCHS"]>limitations["TCHS_lower_limit"]) & (Chiller_data.loc[:, "TCHS"]<limitations["TCHS_upper_limit"])
    mask_TCH_delta = (TCH_delta>limitations["TCH_delta_lower_limit"]) & (TCH_delta<limitations["TCH_delta_upper_limit"])
    # Related to cooling load (RT), energy consumption (KW), and energy performance (KW/RT)
    mask_RT = (Chiller_data.loc[:, "RT_cal"]>limitations["RT_lower_limit"]) & (Chiller_data.loc[:, "RT_cal"]<limitations["RT_upper_limit"])
    mask_KW = (Chiller_data.loc[:, "KW"]>limitations["KW_lower_limit"]) & (Chiller_data.loc[:, "KW"]<limitations["KW_upper_limit"])
    mask_KWRT = (Chiller_data.loc[:, "KWRT_cal"]>limitations["KWRT_lower_limit"]) & (Chiller_data.loc[:, "KWRT_cal"]<limitations["KWRT_upper_limit"])
    # Mask the filters
    Mask = (mask_TCHS & mask_TCH_delta & mask_KWRT & mask_KW & mask_RT)
    Chiller_data = Chiller_data.loc[Mask,:]
    Chiller_data = Chiller_data.dropna()
    return Chiller_data

data_temp = read_Chiller_data(Chiller_name)
display(data_temp)


# %%
# Create dictionary for data cache
# Selected data columns include TCH, TCW, quantity of chilled water (QCH), cooling load (RT_cal), energy use and chiller performance (KW & KWRT_cal), frequency of cooling tower (CT01_cal), energy use and performance of chiller plant system (sysKW & sysKWRT_cal), chiller name (Chiller), and outdoor wet bulb temperature
dict_chiller_properties = {}
dict_chiller_properties["LoadData"] = pd.DataFrame()
LoadData_list = []
select_variable = ["QCH", "TCHS", "TCHR", "TCWS", "RT_cal", "KW", "KWRT_cal", "CT01_Freq", "sysKW", "sysKWRT_cal", "Chiller", "outdrwetT"]

for Chiller_name in Chiller_list:
    data_temp = read_Chiller_data(Chiller_name)
    data_temp.loc[:, "Chiller"] = Chiller_name
    LoadData_list.append(data_temp.loc[:, select_variable])

dict_chiller_properties["LoadData"] = pd.concat(LoadData_list, axis=0, ignore_index=False)    
dict_chiller_properties["LoadData"].reset_index(drop=False, inplace=True)
display(dict_chiller_properties["LoadData"])


# %%
# Initial analysis for chillers
# Display the chillers' cooling load (RT)
LoadData = dict_chiller_properties["LoadData"].copy()
LoadData["DateTime"] = pd.to_datetime(LoadData["DateTime"])
Load_sum = LoadData.pivot_table(index="DateTime",columns="Chiller",values="RT_cal").sum(axis=1).resample("H").mean()
Load_chillers = LoadData.pivot_table(index="DateTime",columns="Chiller",values="RT_cal").resample("H").mean()
Load_chillers["Sum"] = Load_sum
Load_chillers.reset_index(drop=False, inplace=True)

# Display the chillers' energy consumption (KW)
KW_sum = LoadData.pivot_table(index="DateTime",columns="Chiller",values="KW").sum(axis=1).resample("H").mean()
KW_chillers = LoadData.pivot_table(index="DateTime",columns="Chiller",values="KW").resample("H").mean()
KW_chillers["Sum"] = KW_sum
KW_chillers.reset_index(drop=False, inplace=True)

# Display the system energy consumption of chillers (sysKW)
sysKW_sum = LoadData.pivot_table(index="DateTime",columns="Chiller",values="sysKW").sum(axis=1).resample("H").mean()
sysKW_chillers = LoadData.pivot_table(index="DateTime",columns="Chiller",values="sysKW").resample("H").mean()
sysKW_chillers["Sum"] = sysKW_sum
sysKW_chillers.reset_index(drop=False, inplace=True)

# Display the chillers' energy performance (KW/RT)
KWRT_sum = LoadData.pivot_table(index="DateTime",columns="Chiller",values="KWRT_cal").mean(axis=1).resample("H").mean()
KWRT_chillers = LoadData.pivot_table(index="DateTime",columns="Chiller",values="KWRT_cal").resample("H").mean()
KWRT_chillers["mean"] = KWRT_sum
KWRT_chillers.reset_index(drop=False, inplace=True)

# Display the system energy performance of chillers (sysKW/RT)
sysKWRT_sum = LoadData.pivot_table(index="DateTime",columns="Chiller",values="sysKWRT_cal").mean(axis=1).resample("H").mean()
sysKWRT_chillers = LoadData.pivot_table(index="DateTime",columns="Chiller",values="sysKWRT_cal").resample("H").mean()
sysKWRT_chillers["mean"] = sysKWRT_sum
sysKWRT_chillers.reset_index(drop=False, inplace=True)

# %% 
# Describe the chillers' data
Load_describe = Load_chillers[Chiller_list].describe()
RTavg = Load_describe.loc["mean", Chiller_list].values
RTmax = Load_describe.loc["max", Chiller_list].values
Hour_use = Load_describe.loc["count", Chiller_list].values
Load_describe.rename(index={"count": "Using hours", "mean": "Cooling load mean (RTavg)"}, inplace=True)
Load_describe.loc["Total hours", Chiller_list] = ((Load_chillers.loc[len(Load_chillers)-1, "DateTime"]- Load_chillers.loc[0, "DateTime"]).total_seconds() / 3600)
Load_describe.loc["Chiller usage: Using hours/Total hours (%)", Chiller_list] = (Hour_use / Load_describe.loc["Total hours", Chiller_list].values) * 100
Load_describe.loc["Chiller load rate: RTavg/RTmax (%)", Chiller_list] = (RTavg / RTmax) * 100
Load_describe.loc["Cooling amount (RT*hr)", Chiller_list] = RTavg * Hour_use
Load_describe.drop(["std", "min", "25%", "50%", "75%"], axis=0, inplace=True)

KW_describe = KW_chillers[Chiller_list].describe()
KW_describe.rename(index={"count": "Using hours", "mean": "Energy use mean (KWavg)"}, inplace=True)
KW_describe.drop(["Using hours", "std", "min", "25%", "50%", "75%", "max"], axis=0, inplace=True)

sysKW_describe = sysKW_chillers[Chiller_list].describe()
sysKW_describe.rename(index={"count": "Using hours", "mean": "System energy use mean (sysKWavg)"}, inplace=True)
sysKW_describe.drop(["Using hours", "std", "min", "25%", "50%", "75%", "max"], axis=0, inplace=True)

KWRT_describe = KWRT_chillers[Chiller_list].describe()
KWRT_describe.rename(index={"count": "Using hours", "mean": "Energy performance mean (KWRTavg)"}, inplace=True)
KWRT_describe.drop(["Using hours", "std", "min", "25%", "50%", "75%", "max"], axis=0, inplace=True)

sysKWRT_describe = sysKWRT_chillers[Chiller_list].describe()
sysKWRT_describe.rename(index={"count": "Using hours", "mean": "System energy performance mean (sysKWRTavg)"}, inplace=True)
sysKWRT_describe.drop(["Using hours", "std", "min", "25%", "50%", "75%", "max"], axis=0, inplace=True)

Energy_describe = pd.concat([Load_describe, KW_describe, sysKW_describe, KWRT_describe, sysKWRT_describe ], axis=0)
display(Energy_describe.round(2))


# %%
# Cooling Load Plot
x_targ = Load_chillers["DateTime"]
y_sum = Load_chillers["Sum"]
chiller_plot_data = dict()
for Chiller_name in Chiller_list:
    chiller_plot_data[Chiller_name] = Load_chillers[Chiller_name]

y_sum_avg = round(y_sum.mean(), 4)
y_chillers_avg = round(Load_chillers.iloc[:, range(1, Load_chillers.shape[1]-1)].stack().mean(), 4)
(y_sum_color, y_sum_avg_color) = ("#2CBDFE", "#661D98")
sum_avg_text = "Annual average total cooling load:<br>RT=" + str(y_sum_avg)
chillers_avg_text = "Annual average <br>cooling load of each chiller:<br>RT=" + str(y_chillers_avg)

x_stamp = Timestamp("2021-02-01") # for x position of annotation
fig_targ = "RT_total"
fig = go.Figure()
fig.add_trace(go.Scatter(
    name="RT_total", x=x_targ, y=y_sum,
    mode="lines", line=dict(width=1.5, color=y_sum_color)))
fig.add_trace(go.Scatter(
    name="RT_total_avg", x=x_targ, y=[y_sum_avg]*len(x_targ),
    mode="lines", line=dict(dash="dash", width=2, color=y_sum_avg_color)))
fig.add_annotation(
    x=x_stamp, y=y_sum_avg+50, text=sum_avg_text,
    showarrow=False, yshift=10)
fig.update_layout(
    title = dict(text=F"{fig_targ} timeseries"),
    xaxis = dict(
        showline=True, linewidth=1.2, linecolor="black",
        showticklabels=True, showgrid=True, gridcolor="rgba(230, 230, 230, 1)",
        rangeslider=dict(visible=False)),
    yaxis = dict(
        title=dict(text="Cooling Load (RT)", font=dict(size=12), standoff=0),
        dtick=parameters["RT_cal_ticks"],
        showline=True, linewidth=1.2, linecolor="black",
        showgrid=True, gridcolor="rgba(230, 230, 230, 1)"),
    legend = dict(x=1.02, y=0.5, orientation="v", bordercolor="black", borderwidth=0.5, font=dict(size=8)),
    plot_bgcolor="white", width=700, height=400)
fig.show()

fig_targ = "Chillers' RT"
fig = go.Figure()
for Chiller_name, colors in zip(Chiller_list, ["#2CBDFE", "#00cc99", "#F5B14C"]):
    fig.add_trace(go.Scatter(
        name=Chiller_name, x=x_targ, y=chiller_plot_data[Chiller_name],
        mode="lines", line=dict(width=1.5, color=colors)))

fig.add_trace(go.Scatter(
    name="RT_chillers_avg", x=x_targ, y=[y_chillers_avg]*len(x_targ),
    mode="lines", line=dict(dash="dash", width=2, color="#661D98")))
fig.add_annotation(
    x=x_stamp, y=y_chillers_avg+100, text=chillers_avg_text,
    showarrow=False, yshift=10)
fig.update_layout(
    title = dict(text=F"{fig_targ} timeseries"),
    xaxis = dict(
        showline=True, linewidth=1.2, linecolor="black",
        showticklabels=True, showgrid=True, gridcolor="rgba(230, 230, 230, 1)",
        rangeslider=dict(visible=False)),
    yaxis = dict(
        title=dict(text="Cooling Load (RT)", font=dict(size=12), standoff=0),
        dtick=100,
        showline=True, linewidth=1.2, linecolor="black",
        showgrid=True, gridcolor="rgba(230, 230, 230, 1)"),
    legend = dict(x=1.02, y=0.5, orientation="v", bordercolor="black", borderwidth=0.5, font=dict(size=8)),
    plot_bgcolor="white", width=700, height=400)
fig.show()


# %%
# Energy Performance Plot
x_targ = KWRT_chillers["DateTime"]
y_sum = KWRT_chillers["mean"]
KWRT_plot_data = dict()
for Chiller_name in Chiller_list:
    KWRT_plot_data[Chiller_name] = KWRT_chillers[Chiller_name]

y_sum_avg = round(y_sum.mean(), 4)
y_chillers_avg = round((KW_chillers.loc[:, "Sum"]/Load_chillers.loc[:, "Sum"]).mean(), 4)
(y_sum_color, y_sum_avg_color) = ("#2CBDFE", "#661D98")
sum_avg_text = "Annual average chillers performance:<br>KWRT=" + str(y_sum_avg)
chillers_avg_text = "Annual average <br>performance of each chiller:<br>KWRT=" + str(y_chillers_avg)

fig_targ = "Energy Performance of all chillers"
fig = go.Figure()
fig.add_trace(go.Scatter(
    name="KW/RT_total", x=x_targ, y=y_sum,
    mode="lines", line=dict(width=1.5, color=y_sum_color)))
fig.add_trace(go.Scatter(
    name="KW/RT_total_avg", x=x_targ, y=[y_sum_avg]*len(x_targ),
    mode="lines", line=dict(dash="dash", width=2, color=y_sum_avg_color)))

X_stamp = Timestamp("2021-03-01")
Y_stamp = y_sum_avg+0.4
fig.add_annotation(
    x=X_stamp, y=Y_stamp, text=sum_avg_text,
    showarrow=False, yshift=0.1)
fig.update_layout(
    title = dict(text=F"{fig_targ} timeseries", x=0.04, y=0.83),
    xaxis = dict(
        showline=True, linewidth=1.2, linecolor="black",
        showticklabels=True, showgrid=True, gridcolor="rgba(230, 230, 230, 1)",
        rangeslider=dict(visible=False)),
    yaxis = dict(
        title=dict(text="Energy Performance (KW/RT)", font=dict(size=12), standoff=0),
        dtick=parameters["KWRT_cal_ticks"],
        showline=True, linewidth=1.2, linecolor="black",
        zeroline=True, zerolinewidth=1.2, zerolinecolor="rgba(230, 230, 230, 1)",
        showgrid=True, gridcolor="rgba(230, 230, 230, 1)"),
    legend = dict(x=1.02, y=0.5, orientation="v", bordercolor="black", borderwidth=0.5, font=dict(size=8)),
    plot_bgcolor="white", width=700, height=400)
fig.show()

fig_targ = "Energy Performance of each chiller"
fig = go.Figure()
for Chiller_name, colors in zip(Chiller_list, ["#2CBDFE", "#00cc99", "#F5B14C", "#ff3333"]):
    fig.add_trace(go.Scatter(
        name=Chiller_name, x=x_targ, y=KWRT_plot_data[Chiller_name],
        mode="lines", line=dict(width=1.5, color=colors)))

fig.add_trace(go.Scatter(
    name="KW/RT_chillers_avg", x=x_targ, y=[y_chillers_avg]*len(x_targ),
    mode="lines", line=dict(dash="dash", width=2, color="#661D98")))

X_stamp = Timestamp("2021-03-01")
Y_stamp = y_chillers_avg+0.5
fig.add_annotation(
    x=X_stamp, y=Y_stamp, text=chillers_avg_text,
    showarrow=False, yshift=0.1)
fig.update_layout(
    title = dict(text=F"{fig_targ} timeseries", x=0.04, y=0.83),
    xaxis = dict(
        showline=True, linewidth=1.2, linecolor="black",
        showticklabels=True, showgrid=True, gridcolor="rgba(230, 230, 230, 1)",
        rangeslider=dict(visible=False)),
    yaxis = dict(
        title=dict(text="Energy Performance (KW/RT)", font=dict(size=12), standoff=0),
        dtick=parameters["KWRT_cal_ticks"],
        showline=True, linewidth=1.2, linecolor="black",
        zeroline=True, zerolinewidth=1.2, zerolinecolor="rgba(230, 230, 230, 1)",
        showgrid=True, gridcolor="rgba(230, 230, 230, 1)"),
    legend = dict(x=1.02, y=0.5, orientation="v", bordercolor="black", borderwidth=0.5, font=dict(size=8)),
    plot_bgcolor="white", width=700, height=400)
fig.show()


# %%
# Energy Performance of systems Plot
x_targ = sysKWRT_chillers["DateTime"]
y_sum = sysKWRT_chillers["mean"]
sysKWRT_plot_data = dict()
for Chiller_name in Chiller_list:
    sysKWRT_plot_data[Chiller_name] = sysKWRT_chillers[Chiller_name]

y_sum_avg = round(y_sum.mean(), 4)
y_chillers_avg = round((sysKW_chillers.loc[:, "Sum"]/Load_chillers.loc[:, "Sum"]).mean(), 4)
(y_sum_color, y_sum_avg_color) = ("#2CBDFE", "#661D98")
sum_avg_text = "Annual average chiller plants performance:<br>sysKWRT=" + str(y_sum_avg)
chillers_avg_text = "Annual average <br>performance of each chiller plant:<br>sysKWRT=" + str(y_chillers_avg)

fig_targ = "Energy Performance of all chillers' system"
fig = go.Figure()
fig.add_trace(go.Scatter(
    name="KW/RT_total", x=x_targ, y=y_sum,
    mode="lines", line=dict(width=1.5, color=y_sum_color)))
fig.add_trace(go.Scatter(
    name="KW/RT_total_avg", x=x_targ, y=[y_sum_avg]*len(x_targ),
    mode="lines", line=dict(dash="dash", width=2, color=y_sum_avg_color)))

X_stamp = Timestamp("2021-05-01")
Y_stamp = y_sum_avg+0.6
fig.add_annotation(
    x=X_stamp, y=Y_stamp, text=sum_avg_text,
    showarrow=False, yshift=0.1)
fig.update_layout(
    title = dict(text=F"{field_name} {fig_targ} timeseries", x=0.04, y=0.83),
    xaxis = dict(
        showline=True, linewidth=1.2, linecolor="black",
        showticklabels=True, showgrid=True, gridcolor="rgba(230, 230, 230, 1)",
        rangeslider=dict(visible=False)),
    yaxis = dict(
        title=dict(text="Energy Performance (KW/RT)", font=dict(size=12), standoff=0),
        dtick=parameters["KWRT_cal_ticks"],
        showline=True, linewidth=1.2, linecolor="black",
        zeroline=True, zerolinewidth=1.2, zerolinecolor="rgba(230, 230, 230, 1)",
        showgrid=True, gridcolor="rgba(230, 230, 230, 1)"),
    legend = dict(x=1.02, y=0.5, orientation="v", bordercolor="black", borderwidth=0.5, font=dict(size=8)),
    plot_bgcolor="white", width=700, height=400)
fig.show()

fig_targ = "Energy Performance of each chiller"
fig = go.Figure()
for Chiller_name, colors in zip(Chiller_list, ["#2CBDFE", "#00cc99", "#F5B14C", "#ff3333"]):
    fig.add_trace(go.Scatter(
        name=Chiller_name, x=x_targ, y=sysKWRT_plot_data[Chiller_name],
        mode="lines", line=dict(width=1.5, color=colors)))

fig.add_trace(go.Scatter(
    name="KW/RT_chillers_avg", x=x_targ, y=[y_chillers_avg]*len(x_targ),
    mode="lines", line=dict(dash="dash", width=2, color="#661D98")))

X_stamp = Timestamp("2021-05-01")
Y_stamp = y_chillers_avg+0.8
fig.add_annotation(
    x=X_stamp, y=Y_stamp, text=chillers_avg_text,
    showarrow=False, yshift=0.1)
fig.update_layout(
    title = dict(text=F"{field_name} {fig_targ} timeseries", x=0.04, y=0.83),
    xaxis = dict(
        showline=True, linewidth=1.2, linecolor="black",
        showticklabels=True, showgrid=True, gridcolor="rgba(230, 230, 230, 1)",
        rangeslider=dict(visible=False)),
    yaxis = dict(
        title=dict(text="Energy Performance (KW/RT)", font=dict(size=12), standoff=0),
        dtick=parameters["KWRT_cal_ticks"],
        showline=True, linewidth=1.2, linecolor="black",
        zeroline=True, zerolinewidth=1.2, zerolinecolor="rgba(230, 230, 230, 1)",
        showgrid=True, gridcolor="rgba(230, 230, 230, 1)"),
    legend = dict(x=1.02, y=0.5, orientation="v", bordercolor="black", borderwidth=0.5, font=dict(size=8)),
    plot_bgcolor="white", width=700, height=400)
fig.show()


# %%
# Import weather data
Weather_df = pd.read_csv(os.path.join(DATA_PATH, weather_data), index_col=0)
Weather_df["DateTime"] = pd.to_datetime(Weather_df["Date"].astype(str) + " " + Weather_df["Hour"].astype(str)+ ":00")
Weather_df.set_index("DateTime", inplace=True)
display(Weather_df.head())


# %%
# Merge the chillers' data and weather data
processedData_RT = LoadData.pivot_table(index="DateTime",columns="Chiller",values="RT_cal").sum(axis=1).resample("H").mean()
processedData_KW = LoadData.pivot_table(index="DateTime",columns="Chiller",values="KW").sum(axis=1).resample("H").mean()
processedData_sysKW = LoadData.pivot_table(index="DateTime",columns="Chiller",values="sysKW").sum(axis=1).resample("H").mean()
processedData_outdrwetT = LoadData.pivot_table(index="DateTime",columns="Chiller",values="outdrwetT").mean(axis=1).resample("H").mean()
processedData = pd.concat([processedData_RT, processedData_KW, processedData_sysKW, processedData_outdrwetT], axis=1)
processedData.columns = ["RT_cal", "KW", "sysKW", "outdrwetT"]
processedData["KWRT_cal"] = processedData["KW"] / processedData["RT_cal"]
processedData["sysKWRT_cal"] = processedData["sysKW"] / processedData["RT_cal"]
processedData = processedData.merge(Weather_df, left_index=True, right_index=True)
processedData["status"] = (processedData["RT_cal"]>50).astype("float")
display(processedData.head())


# %%
# Weekly profiles of total cooling load
df_plot = processedData.copy()
df_plot["weekday"] = df_plot.index.weekday+1
fig, ax = plt.subplots(1, figsize=(10, 3))
sns.barplot(data=df_plot, x="weekday", y="RT_cal")
plt.title("Weekly profile")
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.show()

# Scatter plot for total cooling load and outdoor temperature
df_plot["outdoorwetT_avg"] = df_plot.loc[:, df_plot.columns.str.contains("wetT")].mean(axis=1)
df_plot["weekday/weekend"] = "weekday"
df_plot.loc[df_plot["weekday"]>5, "weekday/weekend"] = "weekend"
sns.lmplot(x="outdoorwetT_avg", y="RT_cal", hue="weekday/weekend", data=df_plot, order=2, scatter_kws={"alpha":0.2})
plt.title("RT vs wetT distribution")
plt.show()

# Extract the daily max
df_plot = df_plot.resample("D").max()
df_plot["outdoorwetT_avg"] = df_plot.loc[:, df_plot.columns.str.contains("wetT")].mean(axis=1)
df_plot["weekday/weekend"] = "weekday"
df_plot.loc[df_plot["weekday"]>5, "weekday/weekend"] ="weekend"
sns.lmplot(x="outdoorwetT_avg", y="RT_cal", hue="weekday/weekend", data=df_plot, order=2, scatter_kws={"alpha":0.3})
plt.title("Daily $RT_{max}$ vs wetT distribution")
plt.show()


# %%
# Prepare for cooling load prediction (hourly)
df_dataset = processedData[["Temperature", "RH", "outdrwetT", "Precp", "GloblRad", "RT_cal", "status", "KW", "KWRT_cal", "sysKW", "sysKWRT_cal"]].copy()
df_dataset["RT_cal_lag3"] = df_dataset["RT_cal"].shift(3)
df_dataset["RT_cal_lag6"] = df_dataset["RT_cal"].shift(6)
df_dataset["RT_cal_lag12"] = df_dataset["RT_cal"].shift(12)
df_dataset["RT_cal_lag24"] = df_dataset["RT_cal"].shift(24)
df_dataset["Weekday"] = df_dataset.index.weekday 
df_dataset["Hour"] = df_dataset.index.hour + df_dataset.index.minute/60
df_dataset["Precp"] = df_dataset["Precp"].replace("T", 0)
df_dataset.apply(pd.to_numeric)
df_dataset.fillna(0)
df_dataset = df_dataset.dropna()
df_train = df_dataset.loc[:"2021-06-15"]
df_test = df_dataset.loc["2021-06-16":]
df_all = df_dataset.copy()

# Extract variables related to cooling load, and train by lgbm
list_feature = [
    "Temperature", "RH", "outdrwetT", "Precp", "GloblRad",
    "Weekday", "Hour", "status", 
    "RT_cal_lag3", "RT_cal_lag6", "RT_cal_lag12", "RT_cal_lag24"]
pred_targ = "RT_cal"

reg = lgb.LGBMRegressor(n_estimators=50).fit(df_train[list_feature].values, df_train[pred_targ].values)
R2_train = round(reg.score(df_train[list_feature].values, df_train[pred_targ].values),4)
R2_test = round(reg.score(df_test[list_feature].values, df_test[pred_targ].values),4)
R2_all = round(reg.score(df_all[list_feature].values, df_all[pred_targ].values),4)

# View and plot the training results
print(F"R2_train: {str(R2_train)}")
print(F"R2_test: {str(R2_test)}")

df_dataset.loc[df_test.index, F"{pred_targ}_pred"] = reg.predict(df_test[list_feature].values)
df_dataset.loc[:, F"{pred_targ}_pred_all"] = reg.predict(df_all[list_feature].values)

x_targ, y_targ, y_targ_pred = df_dataset.index, df_dataset[pred_targ], df_dataset[F"{pred_targ}_pred"]
y_targ_color, y_targ_pred_color = ["#2CBDFE", "#F5B14C"] 
fig = go.Figure()
fig.add_trace(go.Scatter(
    name=pred_targ, x=x_targ, y=y_targ,
    mode="lines", line=dict(width=1.5, color=y_targ_color)))
fig.add_trace(go.Scatter(
    name=F"{pred_targ}_pred", x=x_targ, y=y_targ_pred,
    mode="lines", line=dict(width=1.5, color=y_targ_pred_color)))
fig.update_layout(
    title = dict(text=F"{pred_targ} prediction result", x=0.04, y=0.83),
    xaxis = dict(
        showline=True, linewidth=1.2, linecolor="black",
        showticklabels=True, showgrid=True, gridcolor="rgba(230, 230, 230, 1)",
        rangeslider=dict(visible=False)),
    yaxis = dict(
        title=dict(text="RT (KW)", font=dict(size=12), standoff=0),
        dtick=200,
        showline=True, linewidth=1.2, linecolor="black",
        showgrid=True, gridcolor="rgba(230, 230, 230, 1)"),
    legend = dict(x=1.02, y=0.5, orientation="v", bordercolor="black", borderwidth=0.5, font=dict(size=8)),
    plot_bgcolor="white", width=700, height=400)
fig.show()


# %%
# Analysis for cooling load prediction (daily)
daily_max_df_dataset = df_dataset[[pred_targ, F"{pred_targ}_pred", F"{pred_targ}_pred_all"]].resample("D").max()
x_targ_daymax, y_targ_daymax, y_targ_pred_daymax = daily_max_df_dataset.index, daily_max_df_dataset[pred_targ], daily_max_df_dataset[F"{pred_targ}_pred"]
R2_daymax = round(r2_score(daily_max_df_dataset.dropna().iloc[:, 0], daily_max_df_dataset.dropna().iloc[:, 1]),4)
R2_daymax_all = round(r2_score(daily_max_df_dataset.iloc[:, 0].dropna(), daily_max_df_dataset.iloc[:, 2].dropna()),4)

fig = go.Figure()
fig.add_trace(go.Scatter(
    name=F"{pred_targ}_daymax", x=x_targ_daymax, y=y_targ_daymax,
    mode="lines", line=dict(width=2, color=y_targ_color)))
fig.add_trace(go.Scatter(
    name=F"{pred_targ}_pred_daymax", x=x_targ_daymax, y=y_targ_pred_daymax,
    mode="lines", line=dict(width=2, color=y_targ_pred_color)))
fig.update_layout(
    title = dict(text=F"{pred_targ}_daymax prediction result (r<sup>2</sup>={R2_daymax})", x=0.04, y=0.83),
    xaxis = dict(
        showline=True, linewidth=1.2, linecolor="black",
        showticklabels=True, showgrid=True, gridcolor="rgba(230, 230, 230, 1)",
        rangeslider=dict(visible=False)),
    yaxis = dict(
        title=dict(text="RT (KW)", font=dict(size=12), standoff=0),
        dtick=200,
        showline=True, linewidth=1.2, linecolor="black",
        showgrid=True, gridcolor="rgba(230, 230, 230, 1)"),
    legend = dict(x=1.02, y=0.5, orientation="v", bordercolor="black", borderwidth=0.5, font=dict(size=8)),
    plot_bgcolor="white", width=700, height=400)
fig.show()


# %%
# Add the df_dataset (hourly predictions) into LoadData (raw data cache), and save
rt_chiller = LoadData.pivot_table(index="DateTime",columns="Chiller",values="RT_cal").resample("H").mean()
ct_freq_chiller = LoadData.pivot_table(index="DateTime",columns="Chiller",values="CT01_Freq").resample("H").mean()
ct_freq_chiller["CT_Freq_mean"] = ct_freq_chiller.mean(axis=1)
tcws_chiller = LoadData.pivot_table(index="DateTime",columns="Chiller",values="TCWS").resample("H").mean()
tcws_chiller["TCWS_max"] = tcws_chiller.max(axis=1)
qch_chiller = LoadData.pivot_table(index="DateTime",columns="Chiller",values="QCH").resample("H").mean()
qch_chiller["QCH_mean"] = qch_chiller.mean(axis=1)
df_dataset = df_dataset.merge(rt_chiller, left_index=True, right_index=True)
df_dataset = df_dataset.merge(ct_freq_chiller[["CT_Freq_mean"]], left_index=True, right_index=True)
df_dataset = df_dataset.merge(tcws_chiller[["TCWS_max"]], left_index=True, right_index=True)
df_dataset = df_dataset.merge(qch_chiller[["QCH_mean"]], left_index=True, right_index=True)
df_dataset = df_dataset.fillna(0)
display(df_dataset)


# %%
# Save the data
df_dataset.reset_index().to_csv(os.path.join(DATA_PATH, "prediction_result_sys.csv"), index=False)
