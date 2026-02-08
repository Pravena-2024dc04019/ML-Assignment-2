**a. Problem Statement**


The hospitality industry faces significant revenue loss due to booking cancellations. 
Last-minute cancellations make it difficult for hotels to resell rooms, leading to inventory inefficiency and lost income. 
The goal of this project is to develop a machine learning solution that predicts the likelihood of a booking being canceled based on customer profiles and reservation details. 


By identifying high-risk bookings in advance, hotel management can -: 
*   Implement targeted retention strategies (e.g., confirmation calls)
*   Optimize overbooking policies to maintain maximum occupancy.
*   Improve revenue forecasting and resource allocation.

This is a Binary Classification problem where the target is to predict if a booking status will be Canceled (1) or Not_Canceled (0).




**b. Dataset description**

No. of Features: 19

No. of Instances: 36275

Features descrition -:
*   Booking_ID - Unique identifier for each booking
*   no_of_adults - Number of adults
*   no_of_children - Number of children
*   no_of_weekend_nights - Number of weekend nights (Sat/Sun)
*   no_of_week_nights - Number of week nights (Mon-Fri)
*   type_of_meal_plan - Type of meal plan booked
*   required_car_parking - Whether a car parking space is required
*   room_type_reserved - Type of room reserved
*   lead_time - Days between booking and arrival
*   arrival_year - Year of arrival date
*   arrival_month - Month of arrival date
*   arrival_date - Day of arrival date
*   market_segment_type - Market segment designation
*   repeated_guest - Is the customer a repeated guest?
*   no_of_previous_cancellations - Number of previous cancellations
*   no_of_previous_bookings_not_canceled - Number of previous bookings not canceled
*   avg_price_per_room - Average price per day of the reservation
*   no_of_special_requests - Total number of special requests
*   booking_status - Target Variable : Whether the booking was canceled or not


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





**c. Models used**


| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|----------|
| Logistic Regression | 0.805651 | 0.857053 |	0.824654 | 0.899979 |	0.860672 |	0.547185 |
| Decision Tree |
| KNN | 
| Naive Bayes | 
| Random Forest (Ensemble) | 
| XGBoost (Ensemble) | 

