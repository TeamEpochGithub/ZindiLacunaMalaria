import pandas as pd
import numpy as np

def save_with_negs(preds: pd.DataFrame, output_path: str):
    # Keep only necessary columns
    columns = ['Image_ID', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    preds = preds[columns]

    # Read negative predictions
    neg = pd.read_csv('/home/epochvpc6/PycharmProjects/ZindiLacunaMalaria/data/neg_preds.csv')

    # Create formatted entries for NEG predictions
    neg_entries = []
    for _, row in neg[neg['prediction'] == 'NEG'].iterrows():
        neg_entries.append({
            'Image_ID': row['file_name'],
            'class': 'NEG',
            'confidence': 1.0,
            'xmin': 0,
            'ymin': 0,
            'xmax': 0,
            'ymax': 0
        })

    # Convert NEG entries to DataFrame
    neg_formatted = pd.DataFrame(neg_entries)

    # Filter ensembled predictions to only include NON_NEG cases
    nonneg = neg[neg['prediction'] == 'NON_NEG']
    preds = preds[preds['Image_ID'].isin(nonneg['file_name'])]

    # Combine ensembled predictions with formatted NEG entries
    final_predictions = pd.concat([preds, neg_formatted])

    # Save results with specific columns
    final_predictions.to_csv(output_path, index=False)


def add_negs_to_submission(df, neg_csv, test_csv):
    neg_df = pd.read_csv(neg_csv)
    test_df = pd.read_csv(test_csv)
    neg_ids = neg_df.loc[neg_df['class'] == 'NEG', 'Image_ID'].unique()
    test_ids = test_df['Image_ID'].unique()
    neg_ids = np.intersect1d(neg_ids, test_ids)
    df = df[~df['Image_ID'].isin(neg_ids)]
    neg_rows = []
    for image_id in neg_ids:
        neg_row = {
            'Image_ID': image_id,
            'class': 'NEG',
            'confidence': 1,
            'ymin': 0,
            'xmin': 0,
            'ymax': 0,
            'xmax': 0
        }
        neg_rows.append(neg_row)
    neg_rows_df = pd.DataFrame(neg_rows)
    df = pd.concat([df, neg_rows_df], ignore_index=True)
    columns = ['Image_ID', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    df = df[columns]

    return df

def add_zero_conf_troph(df,test_csv):
    """This function adds a trophozoite to an image with no preds and not being a NEG image"""
    df = df.copy()
    test_df = pd.read_csv(test_csv)
    test_ids = test_df['Image_ID'].unique()
    pred_ids = df['Image_ID'].unique()
    new_ids = set(test_ids) - set(pred_ids)
    for image_id in new_ids:
        new_row = {
            'Image_ID': image_id,
            'class': 'Trophozoite',
            'confidence': 0,
            'xmin': 0,
            'ymin': 0,
            'xmax': 0,
            'ymax': 0
        }
        df = df.append(new_row, ignore_index=True)
    return df
