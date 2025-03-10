import pandas as pd
import re

# 读取文本文件  | "./sta_result/wilcoxon/txt/mlp_t-3", "r"
# 读取文本文件
classifier = 'tree'
timeframe = 't-5'
with open("wilcoxon/sta_result/wilcoxon/txt/" + classifier + "_" + timeframe, "r") as file:
    content = file.read()

# 正则表达式匹配每组数据
pattern = re.compile(
    r"Ours VS (.*?)\.\n"  # 对比方法
    r"F2\np_value: (.*?) R\+: (.*?) R-: (.*?)\n"  # F2 指标
    r"AUC\np_value: (.*?) R\+: (.*?) R-: (.*?)\n"  # AUC 指标
    r"G-mean\np_value: (.*?) R\+: (.*?) R-: (.*?)\n"  # G-mean 指标
    r"J\np_value: (.*?) R\+: (.*?) R-: (.*?)\n"  # G-mean 指标
    r"-+",  # 分隔符
    re.DOTALL
)

# 解析数据
data = []
for match in pattern.finditer(content):
    method = match.group(1)  # 对比方法
    f2_p_value, f2_r_plus, f2_r_minus = match.group(2), match.group(3), match.group(4)  # F2
    auc_p_value, auc_r_plus, auc_r_minus = match.group(5), match.group(6), match.group(7)  # AUC
    gmean_p_value, gmean_r_plus, gmean_r_minus = match.group(8), match.group(9), match.group(10)  # G-mean
    j_value, j_plus, j_minus = match.group(11), match.group(12), match.group(13)  # G-mean

    # 将数据添加到列表中
    data.append([
        method,
        f2_p_value, f2_r_plus, f2_r_minus,
        auc_p_value, auc_r_plus, auc_r_minus,
        gmean_p_value, gmean_r_plus, gmean_r_minus,
        j_value, j_plus, j_minus
    ])

# 创建 DataFrame
columns = [
    "Method",
    "F2 p-value", "F2 R+", "F2 R-",
    "AUC p-value", "AUC R+", "AUC R-",
    "G-mean p-value", "G-mean R+", "G-mean R-",
    "J p-value", "J R+", "J R-",
]
df = pd.DataFrame(data, columns=columns)

# 保存为 Excel 文件
with pd.ExcelWriter("wilcoxon/sta_result/wilcoxon/excel/" + classifier + "_" + timeframe + ".xlsx", engine="xlsxwriter") as writer:
    # 写入数据
    df.to_excel(writer, index=False, sheet_name="Wilcoxon Test")

    # 获取工作表对象
    workbook = writer.book
    worksheet = writer.sheets["Wilcoxon Test"]

    # 设置表头格式
    header_format = workbook.add_format({
        "bold": True,
        "align": "center",
        "valign": "vcenter",
        "border": 1
    })

    # 设置数据格式
    data_format = workbook.add_format({
        "align": "center",
        "valign": "vcenter",
        "border": 1
    })

    # 应用格式
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    for row_num, row_data in enumerate(df.values, start=1):
        for col_num, value in enumerate(row_data):
            worksheet.write(row_num, col_num, value, data_format)

    # 合并表头
    worksheet.merge_range("B1:D1", "F2", header_format)
    worksheet.merge_range("E1:G1", "AUC", header_format)
    worksheet.merge_range("H1:J1", "G-mean", header_format)
    worksheet.merge_range("K1:M1", "J", header_format)

print("数据已成功保存为 results.xlsx")

