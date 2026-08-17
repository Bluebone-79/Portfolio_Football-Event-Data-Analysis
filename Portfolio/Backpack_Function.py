import pandas as pd
print(f"pandas version: {pd.__version__}")
import numpy as np
print(f"numpy version: {np.__version__}")
import socceraction as soc
print(f"soc version: {soc.__version__}")
import matplotlib.pyplot as plt
from mplsoccer import Pitch

#download the xT model(下载xT模型）
import socceraction.xthreat as xthreat
url_grid = "https://karun.in/blog/data/open_xt_12x8_v1.json"
xT_model = xthreat.load_model(url_grid)
xt_grid = xT_model.xT #xt_grid=(8,12)


#—【caculate the xT value (xT计算函数)】————————
def compute_xt(df, xt_grid):
    rows, cols = xt_grid.shape
    df = df.copy()

    df["start_xT"] = np.nan
    df["end_xT"] = np.nan
    df["xT_added"] = np.nan

    pitch_length = df["Format"].map({
    "7v7": 70,
    "11v11": 105
    })

    pitch_width = df["Format"].map({
    "7v7": 56,
    "11v11": 68
    })

    # Successful passes only(仅包括成功传球）
    success_mask = df["Outcome"] == "Successful"

    # Convert centred pitch coordinates directly into grid indices（将坐标映射进xT模型格子中）
    start_x_bin = np.floor(
        ((df.loc[success_mask, "X"] + pitch_length.loc[success_mask] / 2)
         / pitch_length.loc[success_mask]) * cols
    ).astype(int)

    start_y_bin = np.floor(
        ((df.loc[success_mask, "Y"] + pitch_width.loc[success_mask] / 2) 
         / pitch_width.loc[success_mask]) * rows
    ).astype(int)

    end_x_bin = np.floor(
        ((df.loc[success_mask, "X2"] + pitch_length.loc[success_mask] / 2) 
         / pitch_length.loc[success_mask]) * cols
    ).astype(int)

    end_y_bin = np.floor(
        ((df.loc[success_mask, "Y2"] + pitch_width.loc[success_mask] / 2) 
         / pitch_width.loc[success_mask]) * rows
    ).astype(int)

    # Prevent edge coordinates from falling outside the grid(防止映射坐标溢出)
    start_x_bin = np.clip(start_x_bin, 0, cols - 1)
    start_y_bin = np.clip(start_y_bin, 0, rows - 1)
    end_x_bin = np.clip(end_x_bin, 0, cols - 1)
    end_y_bin = np.clip(end_y_bin, 0, rows - 1)

    # Assign xT values（映射xT值）
    df.loc[success_mask, "start_xT"] = xt_grid[start_y_bin, start_x_bin]
    df.loc[success_mask, "end_xT"] = xt_grid[end_y_bin, end_x_bin]

    df.loc[success_mask, "xT_added"] = (
        df.loc[success_mask, "end_xT"]
        - df.loc[success_mask, "start_xT"]
    )

    return df

## ----------【define passing zone and score state(划分传球区域和比分状态)】--------------------

# Turn the time into seconds and sort the value.(按时间排序事件）
def time_to_seconds(t):
    minute, second = str(t).split(":")
    return int(minute) * 60 + float(second)


def zone_and_score(df,debug=False):
    df=df.copy()
    
    pitch_length = df["Format"].map({
    "7v7": 70,
    "11v11": 105})

    start_zone_bin = np.floor(
        ((df["X"] + pitch_length / 2)/ pitch_length) * 3).astype(int)

    end_zone_bin = np.floor(
        ((df["X2"] + pitch_length / 2)/ pitch_length) * 3).astype(int)

# Prevent boundary values from falling outside the three zones
    start_zone_bin = np.clip(start_zone_bin, 0, 2)
    end_zone_bin = np.clip(end_zone_bin, 0, 2)

    zone_labels = {
        0: "Defensive",
        1: "Middle",
        2: "Attacking"}

    df["start_zone"] = start_zone_bin.map({
        0: "Defensive",
        1: "Middle",
        2: "Attacking"})

    df["end_zone"] = end_zone_bin.map({
        0: "Defensive",
        1: "Middle",
        2: "Attacking"})

# Identify goals
    team_name=df["Team_name"]
    
    df["team_goal"] = (
        (df["Action Type"] == "Shot") & (df["Outcome"] == "Successful") &
        (df["Team"] == team_name))

    df["opp_goal"] = (
        (df["Action Type"] == "Shot") & (df["Outcome"] == "Successful") &
        (df["Team"] != team_name))
    
# Turn the time into seconds and sort the value.(按时间排序）
    df["time_seconds"]= df["Time"].apply(time_to_seconds)
    df = df.sort_values(["match_id", "time_seconds"]).copy()

#adding score state to each pass(为每个pass添加比分状态）
    df["team_score"] = (
        df.groupby("match_id")["team_goal"].cumsum())

    df["opp_score"] = (
        df.groupby("match_id")["opp_goal"].cumsum())
    
    # On goal rows, use the score immediately BEFORE the goal（调整进球事件的比分，为进球前的状态）
    df.loc[df["team_goal"], "team_score"] -= 1
    df.loc[df["opp_goal"], "opp_score"] -= 1

    # Calculate score state(计算比分情况）
    df["score_diff"] = (
        df["team_score"] - df["opp_score"])

    df["score_state"] = np.select(
        [ df["score_diff"] > 0,
            df["score_diff"] == 0,
            df["score_diff"] < 0],
        ["Leading",
         "Drawing",
         "Trailing"])

    # Remove intermediate calculation columns（去除不必要的计算过程）
    if not debug:
        df = df.drop(
            columns=["team_goal","opp_goal",
                "team_score","opp_score",
                    "time_seconds"])

    return df

#-----define the successul and positive pass[定义成功传球和正xt传球]---------
def caculate_success(df):
    df=df.copy()
#1.mark successful passes
    df["success"] = (
        df["Outcome"] == "Successful"
    ).astype(int)
    
#2.mark positive_xt passes
    df["positive_xt"] = (
        df["xT_added"] > 0
    ).astype(int)
    
#3.define transition（e.g. attcak->attack）
    df["Transition"] = (
    df["start_zone"].astype(str)
    + "->"
    + df["end_zone"].astype(str)
    )
    return df


# draw random samples on pitch to test the xt and transition caculation[随机抽取几个样本，并画成图，方便验算xT计算]
def plot_pass(df, game_format, zones=("Defensive", "Middle", "Attacking"), n=2, seed=22):
    samples = []
    
#Only the passes from the game format we what（只包括指定比赛形式的传球）
    format_mask= df["Format"] == game_format
    format_data = df.loc[format_mask]
    
# take (n) samples from each transition（每个球场区域抽取n个传球）
    for zone in zones:
        temp = format_data[
            (format_data["start_zone"] == zone) |
            (format_data[ "end_zone"] == zone)]
        
        samples.append(
            temp.sample(n, random_state=seed))
        
# additional random samples（另外再抽取n个传球）
    samples.append(
        df.sample(n, random_state=seed))

    df_random_sample=pd.concat(samples)
    

    display(df_random_sample[["Time","Outcome","match_id",
             "X", "Y", "X2", "Y2", 
             "start_xT", "end_xT", "xT_added","positive_xt","success",
            "start_zone", "end_zone","Transition"]] )
    
#draw the picture（画图部分）：
    if game_format == "7v7":
        pitch_length = 70
        pitch_width = 56
    
    elif game_format == "11v11":
        pitch_length = 105
        pitch_width = 68
    
    else:
        raise ValueError(f"Unknown game format: {game_format}")

        
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=pitch_length,
        pitch_width=pitch_width )
    
    fig, ax = pitch.draw(figsize=(10, 7))

    for idx, (_, row) in enumerate(df_random_sample.iterrows(), start=1):
        x1 = row["X"] + pitch_length / 2
        y1 = row["Y"] + pitch_width / 2
        x2 = row["X2"] + pitch_length / 2
        y2 = row["Y2"] + pitch_width / 2

        pitch.arrows(
            x1, y1,
            x2, y2,
            ax=ax,
            width=2,
            headwidth=5,
            headlength=5)
        
        pitch.scatter(x1, y1, ax=ax, s=80, label="Start")
        pitch.scatter(x2, y2, ax=ax, s=80, label="End")
        
       # give each pass an index
        ax.text(x1,y1,
            str(idx),
            fontsize=12,
            ha="center",
            va="center")
        
    # draw xT grid lines（画xT grid格子）
    for x in np.linspace(0, pitch_length, 13):
        ax.axvline(    
            x,
            color="grey",linestyle="--",
            linewidth=0.5,alpha=0.5)
    
    for y in np.linspace(0, pitch_width, 9):
        ax.axhline(     
            y,
            color="grey",linestyle="--",
            linewidth=0.5,alpha=0.5)
        
    plt.show()


