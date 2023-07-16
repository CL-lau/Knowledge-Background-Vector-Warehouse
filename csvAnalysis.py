import numpy as np
import pandas as pd

files = ["promotion_stock_record_log.csv",
         "promotion_base.csv",
         "promotion_category.csv",
         "promotion_dim.csv",
         "promotion_ext_attr.csv",
         "promotion_gift_sku.csv",
         "promotion_predict_plan_detail.csv",
         "promotion_predict_plan_sum.csv",
         "promotion_rebate_agreement.csv",
         "promotion_restriction.csv",
         "promotion_sku_stock_control.csv",
         "promotion_sku_stock_used.csv",
         "promotion_sku.csv"]


def get_min_max(row):
    if row['type'] == 'int64':
        values = row['values'].split(',')
        values = [int(v) for v in values]
        return pd.Series({'min_value': min(values), 'max_value': max(values)})
    else:
        return pd.Series({'min_value': None, 'max_value': None})


def csvAnalysis(files):
    for fileItem in files:
        da = pd.read_csv("../../../Downloads/bcp/" + fileItem)
        df = da
        new_df = pd.DataFrame(columns=['name', 'type', 'values', 'count', 'isEnum'])

        for col_name in df.columns:
            col_data = df[col_name].dropna()
            col_type = col_data.dtype.name
            if col_type == "object":
                col_values = "'" + "', '".join(col_data.unique().astype(str)) + "'"
            else:
                col_values = ','.join(col_data.unique().astype(str))
            col_count = len(col_data.unique())
            col_is_enum = col_count < 20
            new_df = pd.concat([new_df, pd.DataFrame({
                'name': [col_name],
                'type': [col_type],
                'values': [col_values],
                'count': [col_count],
                'isEnum': [col_is_enum]
            })], ignore_index=True)

        # 调整列的顺序
        new_df = new_df[['name', 'type', 'values', 'count', 'isEnum']]

        new_df[['min_value', 'max_value']] = new_df.apply(get_min_max, axis=1)
        new_df['type'] = new_df['type'].replace('int64', 'int')
        new_df['type'] = new_df['type'].replace('object', 'String')
        new_df.to_csv("../fund/fund-service/src/main/java/com/sankuai/groceryqa/fund/service/script/" + fileItem,
                      index=False)


csvAnalysis(files)