***a. Problem Statement***


The hospitality industry faces significant revenue loss due to booking cancellations. 
Last-minute cancellations make it difficult for hotels to resell rooms, leading to inventory inefficiency and lost income. 
The goal of this project is to develop a machine learning solution that predicts the likelihood of a booking being canceled based on customer profiles and reservation details. 


By identifying high-risk bookings in advance, hotel management can -: 
*   Implement targeted retention strategies (e.g., confirmation calls)
*   Optimize overbooking policies to maintain maximum occupancy.
*   Improve revenue forecasting and resource allocation.

This is a Binary Classification problem where the target is to predict if a booking status will be Canceled (1) or Not_Canceled (0).

&nbsp;
&nbsp;


***b. Dataset description***

No. of Features: 19

No. of Instances: 36275

Features descrition -:

| Feature Name | Feature Description |
|----------|----------|
| Booking_ID  | Unique identifier for each booking  |
| no_of_adults  | Number of adults  |
| no_of_children  | Number of children  |
| no_of_weekend_nights | Number of weekend nights (Sat/Sun)  |
| no_of_week_nights  | Number of week nights (Mon-Fri)  |
| type_of_meal_plan  | Type of meal plan booked  |
| required_car_parking  | Whether a car parking space is required  |
| room_type_reserved  | Type of room reserved  |
| lead_time  | Days between booking and arrival  |
| arrival_year  | Year of arrival date  |
| arrival_month | Month of arrival date  |
| arrival_date  |  Day of arrival date  |
| market_segment_type  | Market segment designation  |
| repeated_guest  | Is the customer a repeated guest?  |
| no_of_previous_cancellations  | Number of previous cancellations  |
| no_of_previous_bookings_not_canceled  | Number of previous bookings not canceled  |
| avg_price_per_room  | Average price per day of the reservation  |
| no_of_special_requests  | Total number of special requests  |
| booking_status  | Target Variable : Whether the booking was canceled or not  |

&nbsp;
&nbsp;


***c. Models used***


| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|----------|
| Logistic Regression | 0.805651 | 0.857053 |	0.824654 | 0.899979 |	0.860672 |	0.547185 |
| Decision Tree | 0.869883 | 0.859639 |	0.906831 |	0.897086 |	0.901932 |	0.708756 |
| KNN | 0.849897 | 0.904370 |	0.875752 |	0.903079 |	0.889205 |	0.657478 |
| Naive Bayes | 0.445486 |	0.818218 |	0.934043 |	0.181442 |	0.303859 |	0.218619 |
| Random Forest (Ensemble) | 0.905445 |	0.960953 |	0.913728 |	0.947716 |	0.930412 |	0.784373 |
| XGBoost (Ensemble) | 0.892488 |	0.957055 |	0.901166 |	0.942137 |	0.921196 |	0.754177 |

&nbsp;

**Observations on the performance of each model**

| ML Model Name | Observation about model performance |
|-------|----------|
| Logistic Regression | It provides a decent baseline with 80.5% Accuracy. While it has a high Recall (0.89), meaning it finds most cancellations, its lower Precision suggests it might "cry wolf" more often than the ensemble models, incorrectly flagging stable bookings as potential cancellations. |
| Decision Tree | It has 0.70 MCC (Matthews Correlation Coefficient) indicating a strong relationship between predictions and reality, though it lacks the slight "buffer" and refinement provided by the ensemble methods. |
| KNN | It shows a very strong AUC (0.90), meaning it ranks risks well, but its Accuracy (84.9%) and Precision lag slightly behind the tree-based models. It's a solid middle-of-the-road performer but likely computationally heavier than the others. |
| Naive Bayes | It has low accuracy and a very low Recall (0.18) and is failing to identify the vast majority of cancellations. This often happens if the features in our dataset are highly correlated, which violates the "independence" assumption of this model. |
| Random Forest (Ensemble) | It is the **The Top Performer** and strongest model across almost every metric. With an Accuracy of 90.5% and the highest AUC (0.96), it is exceptionally good at distinguishing between guests who will cancel and those who won't. Its high F1 Score (0.93) suggests a great balance between precision and recall. |
| XGBoost (Ensemble) | It performs very similarly to Random Forest. It is highly reliable with an Accuracy of 89.2%. Like Random Forest, its high Recall (0.94) means it is excellent at catching the majority of actual cancellations, which is crucial for hotel revenue management. |



