import pandas as pd
import matplotlib.pyplot as plt

GPT3_MAX_TOKENS = 300_000_000_000 # 300B

data = [
    {
        "name": "B200",
        "toks_per_hr": 2_858_007_564,
        "toks_per_dollar": 477_129_810.35,
        "dollars_per_hr": 5.99
    },
    {
        "name": "H200 SXM",
        "toks_per_hr": 1_793_306_988,
         "toks_per_dollar": 449_450_372.93,
         "dollars_per_hr": 3.99
    },
    {
        "name": "RTX PRO 6000",
        "toks_per_hr": 1_035_986_508,
        "toks_per_dollar": 578_763_412.29,
        "dollars_per_hr": 1.79
    },
    {
        "name": "H100 NVL",
        "toks_per_hr": 943_677_576,
        "toks_per_dollar": 338_235_690.32,
        "dollars_per_hr": 2.79
    },
    {
        "name": "H100 SXM",
        "toks_per_hr": 1_679_183_856,
        "toks_per_dollar": 624_231_916.73,
        "dollars_per_hr": 2.69
    },
    {
        "name": "H100 PCIe",
        "toks_per_hr": 1_042_008_660,
        "toks_per_dollar": 435_986_887.03,
        "dollars_per_hr": 2.39
    },
    {
        "name": "L40",
        "toks_per_hr": 346_844_124,
        "toks_per_dollar": 350_347_600,
        "dollars_per_hr": 0.99
    },
    {
        "name": "L40S",
        "toks_per_hr": 513_506_952,
        "toks_per_dollar": 597_101_106.98,
        "dollars_per_hr": 0.86
    },
    {
        "name": "RTX 6000 Ada",
        "toks_per_hr": 385_761_780,
        "toks_per_dollar": 500_989_324.68,
        "dollars_per_hr": 0.77
    },
    {
        "name": "RTX 5090",
        "toks_per_hr": 707_584_824,
        "toks_per_dollar": 752_749_812.77,
        "dollars_per_hr": 0.94
    },
    {
        "name": "L4",
        "toks_per_hr": 145_521_864,
        "toks_per_dollar": 338_422_939.53,
        "dollars_per_hr": 0.43
    },
    {
        "name": "RTX 4000 Ada",
        "toks_per_hr": 198_372_024,
        "toks_per_dollar": 762_969_323.08,
        "dollars_per_hr": 0.26
    },
    {
        "name": "RTX 2000 Ada",
        "toks_per_hr": 112_895_496,
        "toks_per_dollar": 490_849_982.61,
        "dollars_per_hr": 0.23
    },
    {
        "name": "A100 PCIe",
        "toks_per_hr": 681_816_348,
        "toks_per_dollar": 415_741_675.61,
        "dollars_per_hr": 1.64
    },
    {
        "name": "A100 SXM",
        "toks_per_hr": 752_261_904,
        "toks_per_dollar": 432_334_427.59,
        "dollars_per_hr": 1.74
    },
    {
        "name": "A40",
        "toks_per_hr": 315_402_768,
        "toks_per_dollar": 788_506_920,
        "dollars_per_hr": 0.40
    },
    {
        "name": "RTX A6000",
        "toks_per_hr": 332_798_508,
        "toks_per_dollar": 679_180_628.57,
        "dollars_per_hr": 0.49
    },
    {
        "name": "RTX A5000",
        "toks_per_hr": 265_345_920,
        "toks_per_dollar": 982_762_666.67,
        "dollars_per_hr": 0.27
    },
    {
        "name": "RTX A4500",
        "toks_per_hr": 224_822_052,
        "toks_per_dollar": 899_288_208,
        "dollars_per_hr": 0.25
    },
    {
        "name": "RTX A4000",
        "toks_per_hr": 185_772_348,
        "toks_per_dollar": 743_089_392,
        "dollars_per_hr": 0.25
    },
]

df = pd.DataFrame(data)

latest_gen = {
    "B200", "H200 SXM", "RTX PRO 6000", "H100 NVL",
    "H100 SXM", "H100 PCIe", "L40", "L40S",
    "RTX 6000 Ada", "RTX 5090", "L4",
    "RTX 4000 Ada", "RTX 2000 Ada"
}

df["million_toks_per_hr"] = df["toks_per_hr"] / 1e6
df["million_toks_per_dollar"] = df["toks_per_dollar"] / 1e6
df["generation"] = df["name"].apply(lambda x: "Latest Gen" if x in latest_gen else "Previous Gen")
df["dollars_for_gpt3"] =  GPT3_MAX_TOKENS / df["toks_per_dollar"]
df["hours_for_gpt3"] = GPT3_MAX_TOKENS / df["toks_per_hr"]
df["days_for_gpt3"] = df["hours_for_gpt3"] / 24
df["training_score"] = 1e6 / (df["dollars_for_gpt3"] * df["hours_for_gpt3"])

colors = {
    "Latest Gen": "tab:purple",
    "Previous Gen": "tab:orange"
}

df_sorted_dollars_for_gpt3 = df.sort_values("dollars_for_gpt3")
df_sorted_days_for_gpt3 = df.sort_values("days_for_gpt3", ascending=False)
df_sorted_training_score = df.sort_values("training_score")

fig, axs = plt.subplots(2, 2, figsize=(16, 12))

for gen in df["generation"].unique():
    subset = df[df["generation"] == gen]
    axs[0, 0].scatter(subset["million_toks_per_hr"], subset["million_toks_per_dollar"],
                      s=100, label=gen, color=colors[gen], marker='x')
    for _, row in subset.iterrows():
        axs[0, 0].annotate(row["name"], (row["million_toks_per_hr"], row["million_toks_per_dollar"]),
                           textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)
axs[0, 0].set_xlabel("Millions of tokens per hour (higher is better)")
axs[0, 0].set_ylabel("Millions of tokens per $ (higher is better)")
axs[0, 0].set_title("LLM training throughput (tok/s) vs cost efficiency (tok/$) by GPU model")
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].barh(df_sorted_dollars_for_gpt3["name"], df_sorted_dollars_for_gpt3["dollars_for_gpt3"], color='skyblue')
axs[0, 1].set_xlabel("$ (lower is better)")
axs[0, 1].set_title("Dollars to train on 300B tokens (GPT-3-like) by GPU model")
axs[0, 1].grid(True, axis='x')

axs[1, 0].barh(df_sorted_days_for_gpt3["name"], df_sorted_days_for_gpt3["days_for_gpt3"], color='lightgreen')
axs[1, 0].set_xlabel("Days (lower is better)")
axs[1, 0].set_title("Days to train on 300B tokens (GPT-3-like) by GPU model")
axs[1, 0].grid(True, axis='x')

axs[1, 1].barh(df_sorted_training_score["name"], df_sorted_training_score["training_score"], color='salmon')
axs[1, 1].set_xlabel("1e6 / (dollars for 300B * days for 300B) (higher is better)")
axs[1, 1].set_title("LLM training derivative score (1e6/(cost * time)) by GPU model")
axs[1, 1].grid(True, axis='x')

axs[0, 0].xaxis.set_major_locator(plt.MultipleLocator(250))
axs[0, 0].xaxis.set_minor_locator(plt.MultipleLocator(125))
axs[0, 0].yaxis.set_major_locator(plt.MultipleLocator(100))
axs[0, 0].yaxis.set_minor_locator(plt.MultipleLocator(50))
axs[0, 0].grid(which='minor', alpha=0.2)

axs[0, 1].xaxis.set_major_locator(plt.MultipleLocator(100))
axs[0, 1].xaxis.set_minor_locator(plt.MultipleLocator(50))
axs[0, 1].grid(which='minor', axis='x', alpha=0.2)

axs[1, 0].xaxis.set_major_locator(plt.MultipleLocator(10))
axs[1, 0].xaxis.set_minor_locator(plt.MultipleLocator(5))
axs[1, 0].grid(which='minor', axis='x', alpha=0.2)

axs[1, 1].xaxis.set_major_locator(plt.MultipleLocator(1))
axs[1, 1].xaxis.set_minor_locator(plt.MultipleLocator(0.5))
axs[1, 1].grid(which='minor', axis='x', alpha=0.2)

plt.tight_layout()
plt.show()