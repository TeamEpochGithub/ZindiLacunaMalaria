import pandas as pd


def save_with_negs(preds: pd.DataFrame):
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
    final_predictions.to_csv('ensembled.csv', index=False)