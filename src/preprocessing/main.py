from feature_extractor import FeatureExtractor

# Without chunking
data_dir = "../../data/Composer_Dataset_2"
composers = ["Bach"]
features_df = FeatureExtractor.extract_features_for_multiple_files(data_dir, composers)
print(features_df)
