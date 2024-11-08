import pandas as pd

from postprocessing.ensemble import DualEnsemble

if __name__ == "__main__":
    # Configure ensemble
    ensemble = DualEnsemble(form="wbf", iou_threshold=0.5, conf_threshold=0.1, weights=[1, 1, 1, 1],
                            wbf_reduction='mean')

    # Read prediction files
    file1 = pd.read_csv('predictions_1.csv')
    file2 = pd.read_csv('predictions_2.csv')
    file3 = pd.read_csv('predictions_3.csv')
    file4 = pd.read_csv('predictions_4.csv')

    # Perform ensemble
    files = [file1, file2, file3, file4]
    ensembled = ensemble(files)

    # Keep only necessary columns
    columns = ['Image_ID', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    ensembled = ensembled[columns]

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
    ensembled = ensembled[ensembled['Image_ID'].isin(nonneg['file_name'])]

    # Combine ensembled predictions with formatted NEG entries
    final_predictions = pd.concat([ensembled, neg_formatted])

    # Save results with specific columns
    final_predictions.to_csv('ensembled.csv', index=False)