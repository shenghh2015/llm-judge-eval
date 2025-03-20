from judge_eval.utils.metrics.post_process import check_value_depend_on_all_splits
import matplotlib.pyplot as plt
import numpy as np
import os

##################################
### Visualize Marginal Metrics ###
##################################


def get_mean(series):
  series = series.str.split("|")
  series = series.apply(lambda x: x[-1])
  series = series.str.split("(")
  series = series.apply(lambda x: float(x[0]))

  return series


def get_std(series):
  series = series.str.split("|")
  series = series.apply(lambda x: x[-1])
  series = series.str.split("(")
  series = series.apply(lambda x: x[1])
  series = series.str.split(")")
  series = series.apply(lambda x: float(x[0]))

  return series


def extract_template_name(prompt_series):
  '''Remove the task (summary, hhrlhf) from the template name.'''
  return prompt_series.str.split("_").apply(lambda x: x[0])


def visualize_marginal(df_results, out_folder, metric):
  """
  Visualize Accuracy (Both or Random), Denoised Position Bias, Denoised Length Bias 
  results for the metric grouped by LLM-models.
  """
  # Configs
  df_results = df_results.copy()
  task = '_'.join(df_results.iloc[0]["Prompt"].split("_")[1:])
  metric_name_map = {
      "Accuracy (Both)": "accuracy_both",
      "Accuracy (Random)": "accuracy_random",
      "Position Bias": "position_bias",
      "Length Bias (All Pairs)": "length_bias"
  }
  # Remove the task name and year from templates
  df_results["Prompt"] = df_results["Prompt"].str.split("-").apply(
      lambda x: x[0])
  # Get the mean and std
  series_mean = get_mean(df_results[metric])
  series_std = get_std(df_results[metric])
  df_results = df_results[["Model", "Prompt"]].copy()
  df_results["Mean"] = series_mean
  df_results["Std"] = series_std
  # Group the results by prompts
  means = {
      prompt: df_results.loc[df_results["Prompt"] == prompt, "Mean"].values
      for prompt in df_results["Prompt"].unique()
  }
  stds = {
      prompt: df_results.loc[df_results["Prompt"] == prompt, "Std"].values
      for prompt in df_results["Prompt"].unique()
  }
  # Visualize the result
  # define the x-axis labels and the width of each bar
  labels = df_results["Model"].unique()
  bar_width = 0.08
  # create a numpy array of the x-axis positions for each group of bars
  x_pos = np.arange(len(labels))
  fig, axs = plt.subplots(1, 1, figsize=(15, 10))
  # colored by prompts
  num_prompts = len(df_results["Prompt"].unique())
  colormap = plt.cm.Set3  # LinearSegmentedColormap
  num_colors = min(colormap.N, num_prompts)
  mapcolors = [
      colormap(int(x * colormap.N / num_colors)) for x in range(num_colors)
  ]
  # loop through each group of bars and create a set of bars for each group
  for i, (prompt, values) in enumerate(means.items()):
    # calculate the x-axis position for the current group of bars
    pos = x_pos + (i * bar_width)
    # print(pos, values)
    # create a set of bars for the current group
    axs.bar(pos,
            values,
            yerr=stds[prompt],
            linewidth=4,
            capsize=8,
            error_kw={"elinewidth": 4},
            ecolor='black',
            width=bar_width,
            label=prompt,
            color=mapcolors[i],
            edgecolor='black')
  if metric in ["Accuracy (Both)", "Accuracy (Random)"]:
    axs.set_ylim([0.0, 1.0])
    title = metric
  elif metric == "Position Bias":
    axs.set_ylim([-1.1, 1.4])
    title = "Position Bias"
  elif metric == "Length Bias (All Pairs)":
    axs.set_ylim([0.0, 1.1])
    title = "Length Bias"
  else:
    raise NameError(
        f"Visualization for {metric} is not implemented!")  # for other metrics
  axs.set_xticks(x_pos + ((len(means) - 1) / 2) * bar_width)
  axs.set_xticklabels(labels)
  axs.tick_params(axis='both', which='major', labelsize=30)
  axs.legend(loc="upper center",
             ncol=num_prompts // 2,
             facecolor=(1, 1, 1, 0.3),
             prop={'size': 25})
  axs.set_xlabel('Large Language Models', fontsize=50)
  axs.set_ylabel(title, fontsize=50)
  axs.set_title(title, fontsize=50)

  axs.grid(axis='y', alpha=0.5, linestyle='--', linewidth=3)
  plt.tight_layout()

  # Save the plot
  output_path = os.path.join(
      out_folder, f"viz_results.{task}.{metric_name_map[metric]}.png")
  print(
      f"Write {metric_name_map[metric].replace('_', ' ')} results into {output_path}"
  )
  fig.savefig(output_path, dpi=300)


def visualize_length_bias_acc_long_others(df_results, out_folder):
  '''Another way to visualize length bias: Acc (Human choose long) vs. Acc (Human choose others).'''
  acc_long_results = get_result_matrix(df_results,
                                       "Accuracy_long_denoise (Both)")
  acc_others_results = get_result_matrix(df_results,
                                         "Accuracy_others_denoise (Both)")
  method_names = get_method_names(df_results)

  fig, axs = plt.subplots(1, 1, figsize=(12, 7))
  colormap = plt.cm.tab20b  # LinearSegmentedColormap

  # Colored by llm judges
  llm_set = sorted(
      list(set([name.split("/")[0].strip() for name in method_names])))
  num_llms = len(llm_set)
  num_colors = min(colormap.N, num_llms)
  mapcolors_llm = {
      llm_set[x]: colormap(int(x * colormap.N / num_colors))
      for x in range(num_llms)
  }

  for llm_name in llm_set:
    # Get all the results for this llm
    llm_index = df_results[df_results['Model'] == llm_name].index.tolist()
    llm_acc_long_results = acc_long_results[llm_index, :].ravel()
    llm_acc_others_results = acc_others_results[llm_index, :].ravel()
    axs.scatter(
        llm_acc_others_results,  # acc_others: x-axis
        llm_acc_long_results,  # acc_long: y-axis
        label=llm_name,
        color=mapcolors_llm[llm_name])
  axs.plot(axs.get_xlim(),
           [axs.get_xlim()[0], axs.get_xlim()[1]],
           ls='--',
           c='r',
           label="Acc(Long)=Acc(Others)")
  axs.set_xlabel("Accuracy (Others)")
  axs.set_ylabel("Accuracy (Long)")
  axs.set_title("Accuracy (Long) vs. Accuracy (Others)")
  axs.legend(fontsize=8)
  axs.grid(alpha=0.5, linestyle='--', linewidth=1)

  plt.tight_layout()
  task = '_'.join(df_results.iloc[0]["Prompt"].split("_")[1:])
  output_path = os.path.join(
      out_folder, f"viz_results.{task}.length_bias_acc_long_others.png")
  print(f"Write length bias (long vs. others) results into {output_path}")
  fig.savefig(output_path, dpi=300)


#####################################################################
### Visualize Position and Length Bias with Accuracy Relationship ###
#####################################################################


def get_result_matrix(df_results, metric):
  '''Get all the results from all the methods and splits.'''
  result_matrix = []
  if check_value_depend_on_all_splits(df_results[metric]):
    for _, row in df_results.iterrows():
      result_matrix.append(
          [float(num.strip()) for num in row[metric].split('|')[0].split(',')])
  else:
    for _, row in df_results.iterrows():
      result_matrix.append(float(row[metric].split('(')[0].strip()))

  return np.array(result_matrix)


def get_method_names(df_results):
  '''Get method (LLM + template) names'''
  df_results = df_results.copy()
  df_results["Method"] = df_results.apply(
      lambda x: f"{x['Model']}/{x['Prompt']}", axis=1)
  return df_results["Method"].values


def visualize_position_bias_accuracy(df_results, out_folder):
  '''Visualize the relationship between (absolute) denoised position bias and accuracy.'''
  df_results = df_results.copy()
  acc_results = get_result_matrix(df_results, "Accuracy (Both)")
  bias_results = get_result_matrix(df_results, "Position Bias")
  bias_results = np.abs(bias_results)  # compute absolute value of bias
  method_names = get_method_names(df_results)  # method = llm + template

  fig, axs = plt.subplots(1, 1, figsize=(7, 5))
  colormap = plt.cm.tab20b  # LinearSegmentedColormap
  # markermap = itertools.cycle(['o', 'P', 's', 'v', '^', '<', '>', 'd', 'p', '*', "8"])

  # Colored and marked by LLMs
  llm_set = sorted(
      list(set([name.split("/")[0].strip() for name in method_names])))
  num_llms = len(llm_set)
  num_colors = min(colormap.N, num_llms)
  mapcolors_llm = {
      llm_set[x]: colormap(int(x * colormap.N / num_colors))
      for x in range(num_llms)
  }

  for llm_name in llm_set:
    # Get all the results for this llm
    llm_index = df_results[df_results['Model'] == llm_name].index.tolist()
    llm_acc_results = acc_results[llm_index, :].ravel()
    llm_bias_results = bias_results[llm_index, :].ravel()
    axs.scatter(llm_acc_results,
                llm_bias_results,
                label=llm_name,
                color=mapcolors_llm[llm_name])

  axs.set_xlabel("Accuracy (Both)", fontsize=20)
  axs.set_ylabel("Absolute Position Bias", fontsize=20)
  axs.set_title("Absolute Position Bias vs. Accuracy (Both)", fontsize=20)
  axs.tick_params(axis='both', which='major', labelsize=10)
  axs.legend(fontsize=15)
  # axs.set_ylim([-0.1, 1.1])
  # axs.set_xlim([-0.1, 1.1])
  axs.grid(alpha=0.5, linestyle='--', linewidth=2)

  plt.tight_layout()
  task = '_'.join(df_results.iloc[0]["Prompt"].split("_")[1:])
  output_path = os.path.join(out_folder,
                             f"viz_results.{task}.position_bias_accuracy.png")
  print(f"Write position bias vs. accuracy results into {output_path}")
  fig.savefig(output_path, dpi=300)


def visualize_length_bias_accuracy(df_results, out_folder):
  '''Visualize the relationship between (absolute) denoised length bias and accuracy.'''
  df_results = df_results.copy()
  acc_results = get_result_matrix(df_results, "Accuracy (Both)")
  bias_results = get_result_matrix(df_results, "Length Bias (Splits)")
  bias_results = np.abs(bias_results)  # compute absolute value of bias
  method_names = get_method_names(df_results)  # method = llm + template

  fig, axs = plt.subplots(1, 1, figsize=(7, 5))
  colormap = plt.cm.tab20b  # LinearSegmentedColormap
  # markermap = itertools.cycle(['o', 'P', 's', 'v', '^', '<', '>', 'd', 'p', '*', "8"])

  # Colored and marked by LLMs
  llm_set = sorted(
      list(set([name.split("/")[0].strip() for name in method_names])))
  num_llms = len(llm_set)
  num_colors = min(colormap.N, num_llms)
  mapcolors_llm = {
      llm_set[x]: colormap(int(x * colormap.N / num_colors))
      for x in range(num_llms)
  }

  for llm_name in llm_set:
    # Get all the results for this llm
    llm_index = df_results[df_results['Model'] == llm_name].index.tolist()
    llm_acc_results = acc_results[llm_index, :].ravel()
    llm_bias_results = bias_results[llm_index, :].ravel()
    axs.scatter(llm_acc_results,
                llm_bias_results,
                label=llm_name,
                color=mapcolors_llm[llm_name])

  axs.set_xlabel("Accuracy (Both)", fontsize=20)
  axs.set_ylabel("Length Bias", fontsize=20)
  axs.set_title("Length Bias vs. Accuracy (Both)", fontsize=20)
  axs.legend(fontsize=15)
  #axs.set_ylim([-0.1, 1.1])
  #axs.set_xlim([-0.1, 1.1])
  axs.grid(alpha=0.5, linestyle='--', linewidth=1)

  plt.tight_layout()
  task = '_'.join(df_results.iloc[0]["Prompt"].split("_")[1:])
  output_path = os.path.join(out_folder,
                             f"viz_results.{task}.length_bias_accuracy.png")
  print(f"Write length bias vs. accuracy results into {output_path}")
  fig.savefig(output_path, dpi=300)


def visualize_position_length_bias(df_results, out_folder):
  '''Visualize the relationship between (absolute) position bias and (absolute) length bias.'''
  df_results = df_results.copy()
  position_bias_results = np.abs(get_result_matrix(df_results,
                                                   "Position Bias"))
  length_bias_results = np.abs(
      get_result_matrix(df_results, "Length Bias (Splits)"))
  method_names = get_method_names(df_results)  # method = llm + template

  fig, axs = plt.subplots(1, 1, figsize=(5, 5))
  colormap = plt.cm.tab20b  # LinearSegmentedColormap
  # markermap = itertools.cycle(['o', 'P', 's', 'v', '^', '<', '>', 'd', 'p', '*', "8"])

  # Colored and marked by LLMs
  llm_set = sorted(
      list(set([name.split("/")[0].strip() for name in method_names])))
  num_llms = len(llm_set)
  num_colors = min(colormap.N, num_llms)
  mapcolors_llm = {
      llm_set[x]: colormap(int(x * colormap.N / num_colors))
      for x in range(num_llms)
  }

  for llm_name in llm_set:
    # Get all the results for this llm
    llm_index = df_results[df_results['Model'] == llm_name].index.tolist()
    llm_pos_bias_results = position_bias_results[llm_index, :].ravel()
    llm_len_bias_results = length_bias_results[llm_index, :].ravel()
    axs.scatter(llm_len_bias_results,
                llm_pos_bias_results,
                label=llm_name,
                color=mapcolors_llm[llm_name])

  axs.set_xlabel("Absolute Length Bias")
  axs.set_ylabel("Absolute Position Bias")
  axs.set_title("Absolute Position Bias vs. Absolute Length Bias")
  axs.legend(fontsize=8)
  # axs.set_ylim([-0.1, 1.1])
  axs.grid(alpha=0.5, linestyle='--', linewidth=1)

  plt.tight_layout()
  task = '_'.join(df_results.iloc[0]["Prompt"].split("_")[1:])
  output_path = os.path.join(out_folder,
                             f"viz_results.{task}.position_length_bias.png")
  print(f"Write position bias vs. length bias results into {output_path}")
  fig.savefig(output_path, dpi=300)
