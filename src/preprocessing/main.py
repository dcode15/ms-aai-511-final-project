from feature_extractor import FeatureExtractor

# Without chunking
data_dir = "../../data/Composer_Dataset_2"
composers = ["Bach"]
scalar_features, multidimensional_features = FeatureExtractor.extract_features_for_multiple_files(data_dir, composers)
scalar_features.head()
