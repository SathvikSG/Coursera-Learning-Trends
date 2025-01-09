import csv

# Open the data file and turn it into a list of rows
# Feel free to swap in other datasets if needed
# data=list(csv.reader(open('test1.csv'))) or test2.csv if you're feeling fancy
# for now let's work with 'CourseraDataset-Clean.csv'
data = list(csv.reader(open('CourseraDataset-Clean.csv')))

# Quick peek at the header row (uncomment if you wanna see it)
# print(data[0])

# Print the number of rows in the data (including header)
# print(len(data))

def getcol(num):
    """
    Pulls a whole column from the dataset (ignoring the header row).
    Basically lets you grab a list of values from a specific column.
    """
    col = []
    for i in range(1, len(data)):
        if i == 0:  # Skips the header (duh)
            continue
        col.append(data[i][num])
    return col

# Now let's see which universities get the highest reviews on Coursera
# Column 8 is 'university' and column 12 is 'reviews'
def calculate_average_review(universities, reviews):
    university_review_dict = {}
    # Loop through the universities and their reviews
    for university, review in zip(universities, reviews):
        if university in university_review_dict and review is not None:
            university_review_dict[university].append(review)
        else:
            university_review_dict[university] = [review]

    # Calculate the average review for each university
    average_reviews = {}
    for university, reviews_list in university_review_dict.items():
        average_reviews[university] = sum(reviews_list) / len(reviews_list)

    # Sort universities by average review and keep top 10
    average_reviews = dict(sorted(average_reviews.items(), key=lambda item: item[1], reverse=True))
    return dict(list(average_reviews.items())[:10])

# Next, let’s figure out if course difficulty has any relation with how long it takes to complete.
# Column 2 is 'level' and column 11 is 'duration'
def convert_to_numeric(col):
    """
    Converts column values to floats. 
    If you pass it non-numeric data, it’s probably gonna break things.
    """
    newCol = []
    for value in col:
        newCol.append(float(value))
    return newCol

def calculate_association(levels, durations):
    """
    Find the average time (in hours) it takes to finish a course at each difficulty level.
    """
    association_dict = {}

    for level, duration in zip(levels, durations):
        if level in association_dict and duration is not None:
            association_dict[level].append(duration)
        elif duration is not None:
            association_dict[level] = [duration]

    # Calculate the average duration for each level
    average_durations = {}
    for level, durations_list in association_dict.items():
        average_durations[level] = sum(durations_list) / len(durations_list) if durations_list else None

    return average_durations

# Now let's find which skills on Coursera have the highest ratings.
# Column 5 is 'skills' and column 1 is 'rating'.
def calculate_skill_ratings(skills, ratings):
    skill_rating_dict = {}

    for skill, rating in zip(skills, ratings):
        if skill in skill_rating_dict and rating is not None:
            skill_rating_dict[skill].append(rating)
        else:
            skill_rating_dict[skill] = [rating]

    # Average rating per skill
    average_ratings = {}
    for skill, ratings_list in skill_rating_dict.items():
        average_ratings[skill] = sum(ratings_list) / len(ratings_list)

    # Sort by rating and grab top 5
    return dict(sorted(average_ratings.items(), key=lambda item: item[1], reverse=True)[:5])

# Function 1: Average review per university
review_col = getcol(12)
university_col = getcol(8)
review_col = [float(value) if value.replace('.', '', 1).isdigit() else None for value in review_col]
average_reviews = calculate_average_review(university_col, review_col)

print("The top 10 universities with the highest reviews are:")
for university, avg_review in average_reviews.items():
    print(f"University: {university}, Average review: {avg_review:.2f} reviews")

print("\n")

# Function 2: Association between course difficulty and completion time
level2_col = getcol(2)
duration_col = getcol(11)
duration_col = convert_to_numeric(duration_col)
association_results = calculate_association(level2_col, duration_col)

print("The average time in hours it takes to finish a course for each difficulty level is:")
for level, avg_duration in association_results.items():
    if avg_duration is not None:
        print(f"Level: {level}, Average Duration: {avg_duration:.2f} hours")
    else:
        print(f"Level: {level}, No durations available")

print("\n")

# Function 3: Top 5 skills with the highest ratings
skill_col = getcol(5)
rating_col = getcol(1)
rating_col = convert_to_numeric(rating_col)
ratings = calculate_skill_ratings(skill_col, rating_col)

print("The top 5 skills with the highest ratings are:")
for skill, rating in ratings.items():
    print(f"Skill: {skill}, Rating: {rating:.2f} stars")

print("")
print("")
print("")


#Predictive Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Encoding columns
university_col = getcol(8)
level2_col = getcol(2)
rating_col = convert_to_numeric(getcol(1))

# Initialize the encoders
encoder_university = LabelEncoder()
encoder_level = LabelEncoder()

# Fit the encoders to the dataset values
X_university = encoder_university.fit_transform(university_col)
X_level = encoder_level.fit_transform(level2_col)
y_rating = np.array(rating_col)

# Combine features into a single dataset
X = np.column_stack((X_university, X_level))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_rating, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print("Predictive Analysis Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Example Prediction with Correct Label Transformation
example_university = "Stanford University"
example_level = "Beginner level"  # Ensure this matches the encoded label

# Transform using fitted encoders
example_university_encoded = encoder_university.transform([example_university])[0]
example_level_encoded = encoder_level.transform([example_level])[0]

# Predict
predicted_rating = model.predict([[example_university_encoded, example_level_encoded]])[0]
print(f"\nPredicted rating for a {example_level} course at {example_university}: {predicted_rating:.2f} stars")
