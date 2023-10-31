from typing import Union
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import requests

app = FastAPI()

async def process_json_data(request: Request):
    try:
        json_data = await request.json()
        userID = json_data.get("userID")
        
        access_password = json_data.get("access_password")
        if access_password != "noshInternalPassKey":
            return {"status" : 401 , "message": "Unauthorized"}
        
        api_url = f'https://a5morsuyy6.execute-api.ap-south-1.amazonaws.com/dev/cookings?userId=USER%23{userID}'
        resp = requests.get(api_url)
        user_cookings_data = resp.json()
        
        if not user_cookings_data:
            return {"status" : 404 , "message": "User cookings data not found"}
        return {"status" : 200 , "user_cookings_data" : user_cookings_data, "message": "Data processed successfully"}
        
    except Exception as e:
        print(e)
        
@app.get("/")
async def calc_recommendations(request: Request):
    try:
        response = await process_json_data(request)

        if(response['status'] != 200):
            return response

        else:    
            user_cookings_data = response['user_cookings_data']
            user_cookings_data = [{"timestamp": item["timestamp"], "dish": item["dish"]} for item in user_cookings_data if "timestamp" and "dish" in item]
            user_sorted_cookings_data = sorted(user_cookings_data, key=lambda x: x['timestamp'], reverse=True)

            api_url = "https://a5morsuyy6.execute-api.ap-south-1.amazonaws.com/dev/dish-features"
            resp = requests.get(api_url)
            dish_features_data = resp.json()
            df= pd.DataFrame(dish_features_data)
            valid_dish_ids = set(df['DishID'])
            
            unique_dish_ids = set()
            valid_cookings = []
            for item in user_sorted_cookings_data:
                if len(valid_cookings) < 3:
                    dish_id = item["dish"]
                    if dish_id not in unique_dish_ids and dish_id in valid_dish_ids:
                        unique_dish_ids.add(dish_id)
                        valid_cookings.append(dish_id)
                else:
                    break

            if not valid_cookings:
                return {"status" : 400 , "message": "No valid dish IDs in the user cokkings"}
            
            df['combined_text'] = df['Course'] + ' ' + df['Cuisine'] + ' ' + df['Type'] + ' ' + df['Consistency'] + ' ' + df['MainIngredient'] + ' ' + df['Flavor1'] + ' ' + df['Flavor2']        
            vectorizer = TfidfVectorizer()  # Use TF-IDF to vectorize the text data
            tfidf_matrix = vectorizer.fit_transform(df['combined_text'])  
            
            user_cooked_indices = df[df['DishID'].isin(valid_cookings)].index   # Get the indices of the user's cooked dishes
            mean_cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)   # Calc the mean cosine similarity with user's cooked dishes
            similar_dish_indices = mean_cosine_sim[user_cooked_indices].mean(axis=0).argsort()[::-1] # Get the indices of dishes sorted by cosine similarity
            similar_dish_indices = [idx for idx in similar_dish_indices if idx not in user_cooked_indices] # Exclude user's cooked dishes from recommendations
            
            max_breakfast_dishes = 2
            breakfast_count = 0
            filtered_recommended_indexes = []

            for idx in similar_dish_indices:
                if df.iloc[idx]['Course'] == 'Breakfast':
                    if breakfast_count < max_breakfast_dishes:
                        filtered_recommended_indexes.append(idx)
                        breakfast_count += 1
                else:
                    filtered_recommended_indexes.append(idx)
            recommended_dishes = df.iloc[filtered_recommended_indexes[:5]]  # Get the top 5 recommendations
        
            return {
                "status" : 200 , 
                "recommended_dishes" : recommended_dishes['DishID'].tolist(), 
                "message": f"Recommended Dishes based on Mean Preferences of {', '.join(valid_cookings)}"
                }

    except Exception as error:
        return {"status" : 400 , "message": str(error)}
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)