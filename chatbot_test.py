import os
import json
import pandas as pd
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import List, Dict, Any, Tuple, Optional
import openai
from IPython.display import display, Markdown
from scipy import spatial

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY", "your-api-key")
# Model configuration
GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

#########################################
# 1. Data Loading and Preprocessing
#########################################

class DataProcessor:
    def __init__(self, parking_path: str = "./places-data/parking.json", 
                 sushi_path: str = "./places-data/sushi.json"):
        """Initialize data processor, load parking and sushi restaurant data"""
        self.parking_path = parking_path
        self.sushi_path = sushi_path
        self.parking_data = None
        self.sushi_data = None
        self.place_index = {}
        self.place_embeddings = {}
        
    def load_data(self) -> None:
        """Load raw data files"""
        try:
            with open(self.parking_path, "r", encoding="utf-8") as f:
                self.parking_data = json.load(f)
            with open(self.sushi_path, "r", encoding="utf-8") as f:
                self.sushi_data = json.load(f)
            print(f"Loaded {len(self.parking_data)} parking facilities and {len(self.sushi_data)} sushi restaurants")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def create_index(self) -> None:
        """Create quick index for data"""
        if not self.parking_data or not self.sushi_data:
            self.load_data()
        
        # Create index for parking facilities
        self.place_index["parking"] = {}
        for i, place in enumerate(self.parking_data):
            title = place["place"]["title"]
            self.place_index["parking"][title] = {
                "index": i,
                "data": place,
                "distance": place["distance"],
                # Extract key fields for quick filtering
                "keywords": self._extract_keywords(place, "parking")
            }
        
        # Create index for sushi restaurants
        self.place_index["sushi"] = {}
        for i, place in enumerate(self.sushi_data):
            title = place["place"]["title"]
            self.place_index["sushi"][title] = {
                "index": i,
                "data": place,
                "distance": place["distance"],
                # Extract key fields for quick filtering
                "keywords": self._extract_keywords(place, "sushi")
            }
    
    def _extract_keywords(self, place: Dict[str, Any], place_type: str) -> Dict[str, Any]:
        """Extract key fields from place data for quick filtering"""
        keywords = {
            "title": place["place"]["title"].lower(),
            "address": place["place"]["formattedAddress"].lower(),
            "distance": place["distance"],
            "is_open": place["place"].get("businessHours", {}).get("status") == "OPEN"
        }
        
        # Parking specific fields
        if place_type == "parking":
            keywords["price_2h"] = self._extract_parking_price(place, "2")
            keywords["spots"] = place["place"].get("parking", {}).get("spotsNumber", 0)
            keywords["height"] = place["place"].get("parking", {}).get(
                "parkingDimensionRestriction", {}).get("height", 0)
        
        # Sushi restaurant specific fields
        elif place_type == "sushi":
            keywords["rating"] = place["place"].get("reviewSummary", {}).get("averageRating", 0)
            keywords["price_level"] = place["place"].get("commercial", {}).get(
                "priceSummary", {}).get("priceRangeLevel", 0)
            
            # Extract cuisine types
            food_types = []
            if "foodTypes" in place["place"]:
                for food in place["place"]["foodTypes"]:
                    food_types.append(food["name"].lower())
            keywords["food_types"] = food_types
        
        return keywords
    
    def _extract_parking_price(self, place: Dict[str, Any], hours: str) -> float:
        """Try to extract price for specific number of hours from parking data"""
        try:
            if "commercial" in place["place"] and "priceStructured" in place["place"]["commercial"]:
                prices = place["place"]["commercial"]["priceStructured"]
                if "preComputedPrices" in prices:
                    for price_item in prices["preComputedPrices"]:
                        if price_item.get("quantity") == int(hours) and price_item.get("unit") == "hour":
                            return price_item.get("price", 0)
                
                if "listPrices" in prices:
                    for price_item in prices["listPrices"]:
                        if price_item.get("quantity") == int(hours) and price_item.get("unit") == "hours":
                            return price_item.get("price", 0)
            return 0
        except:
            return 0
    
    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def create_embeddings(self) -> None:
        """Create embedding vectors for all places"""
        if not self.place_index:
            self.create_index()
        
        # Create embeddings for parking facilities
        self.place_embeddings["parking"] = {}
        for title, place_info in self.place_index["parking"].items():
            # Create rich text description for embedding
            description = self._create_embedding_text(place_info["data"], "parking")
            response = openai.Embedding.create(
                input=description,
                model=EMBEDDING_MODEL
            )
            self.place_embeddings["parking"][title] = response["data"][0]["embedding"]
        
        # Create embeddings for sushi restaurants
        self.place_embeddings["sushi"] = {}
        for title, place_info in self.place_index["sushi"].items():
            # Create rich text description for embedding
            description = self._create_embedding_text(place_info["data"], "sushi")
            response = openai.Embedding.create(
                input=description,
                model=EMBEDDING_MODEL
            )
            self.place_embeddings["sushi"][title] = response["data"][0]["embedding"]
    
    def _create_embedding_text(self, place: Dict[str, Any], place_type: str) -> str:
        """Create rich text description for generating embeddings"""
        p = place["place"]
        base_text = f"Name: {p['title']}. Address: {p['formattedAddress']}. Distance: {place['distance']} meters."
        
        # Add business hours
        hours_text = ""
        if "businessHours" in p and "formattedHours" in p["businessHours"]:
            hours_text = f"Business hours: {', '.join(p['businessHours']['formattedHours'])}."
        base_text += f" {hours_text}"
        
        # Add parking specific information
        if place_type == "parking":
            parking_info = p.get("parking", {})
            commercial = p.get("commercial", {})
            
            # Add spots information
            spots_text = ""
            if "spotsNumber" in parking_info:
                spots_text = f"Total spots: {parking_info['spotsNumber']}."
            if "freeSpotsNumber" in parking_info:
                spots_text += f" Free spots: {parking_info['freeSpotsNumber']}."
            base_text += f" {spots_text}"
            
            # Add price information
            price_text = ""
            if "priceSummary" in commercial:
                price_text = f"Price: {commercial['priceSummary'].get('priceSummaryText', '')}."
            base_text += f" {price_text}"
            
            # Add dimension restrictions
            dimension_text = ""
            if "parkingDimensionRestriction" in parking_info:
                dim = parking_info["parkingDimensionRestriction"]
                if "height" in dim:
                    dimension_text = f"Height limit: {dim['height']} meters."
                if "width" in dim:
                    dimension_text += f" Width limit: {dim['width']} meters."
            base_text += f" {dimension_text}"
        
        # Add sushi restaurant specific information
        elif place_type == "sushi":
            # Add rating information
            rating_text = ""
            if "reviewSummary" in p:
                review = p["reviewSummary"]
                rating_text = f"Rating: {review.get('averageRating', 0)}/5, {review.get('reviewCount', 0)} reviews."
            base_text += f" {rating_text}"
            
            # Add price level
            price_text = ""
            if "commercial" in p and "priceSummary" in p["commercial"]:
                price_summary = p["commercial"]["priceSummary"]
                if "priceRangeText" in price_summary:
                    price_text = f"Price level: {price_summary['priceRangeText']}."
            base_text += f" {price_text}"
            
            # Add cuisine information
            food_text = ""
            if "foodTypes" in p:
                food_types = [food["name"] for food in p["foodTypes"]]
                food_text = f"Cuisines: {', '.join(food_types)}."
            base_text += f" {food_text}"
        
        return base_text
    
    def preprocess_all(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, List[float]]]]:
        """Execute all data preprocessing steps and return index and embeddings"""
        self.load_data()
        self.create_index()
        self.create_embeddings()
        return self.place_index, self.place_embeddings

#########################################
# 2. Intent Recognition and Query Processing
#########################################

class IntentProcessor:
    def __init__(self):
        """Initialize intent processor"""
        pass
    
    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def detect_intent(self, query: str) -> Dict[str, Any]:
        """Determine user's query intent"""
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "Identify whether the user is querying about parking facilities or sushi restaurants. If unclear, analyze the more likely option."},
                {"role": "user", "content": query}
            ],
            functions=[{
                "name": "determine_intent",
                "description": "Determine the main intent of the user query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string", 
                            "enum": ["parking", "sushi", "unknown"],
                            "description": "Main category of user query, parking for parking facilities, sushi for sushi restaurants, unknown if can't determine"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level of intent determination, between 0-1"
                        },
                        "query_type": {
                            "type": "string",
                            "enum": ["recommendation", "details", "comparison", "availability", "other"],
                            "description": "Query type: recommendation, details query, comparison analysis, availability query or other"
                        },
                        "specific_attributes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific attributes mentioned in the query, such as price, distance, rating, etc."
                        }
                    },
                    "required": ["intent", "confidence", "query_type"]
                }
            }],
            function_call={"name": "determine_intent"},
            temperature=0.1
        )
        
        function_args = response.choices[0].message.function_call.arguments
        return json.loads(function_args)
    
    def is_simple_query(self, query: str, intent_info: Dict[str, Any]) -> bool:
        """Determine if this is a simple query that can be processed directly through filtering and sorting"""
        # Recommendation queries are usually simple
        if intent_info["query_type"] == "recommendation":
            return True
        
        # If specific attributes or comparison keywords are mentioned, it might be a complex query
        complex_indicators = [
            "compare", "difference", "better", "best", "recommend", "vs", "versus"
        ]
        
        for indicator in complex_indicators:
            if indicator in query.lower():
                return False
        
        # Default to simple query
        return True

#########################################
# 3. Information Retrieval
#########################################

class InformationRetriever:
    def __init__(self, place_index: Dict[str, Dict[str, Any]], 
                 place_embeddings: Dict[str, Dict[str, List[float]]]):
        """Initialize information retriever"""
        self.place_index = place_index
        self.place_embeddings = place_embeddings
    
    def direct_filter_search(self, query: str, intent: str, 
                             context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform simple query through direct filtering and sorting"""
        results = []
        places = self.place_index.get(intent, {})
        
        # Extract key fields for filtering
        for title, place_info in places.items():
            score = self._calculate_relevance_score(query, place_info, intent)
            if score > 0:
                results.append({
                    "title": title,
                    "data": place_info["data"],
                    "relevance_score": score
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:5]  # Return top 5 results
    
    def _calculate_relevance_score(self, query: str, place_info: Dict[str, Any], intent: str) -> float:
        """Calculate relevance score between place and query"""
        score = 0.0
        keywords = place_info["keywords"]
        query_lower = query.lower()
        
        # Base scoring: default sorting by distance
        # Convert distance to a 0-1 score, closer = higher score
        base_distance_score = 1.0 / (1.0 + keywords["distance"] / 500.0)  # 500m as reference distance
        score += base_distance_score * 0.7  # Distance weight 0.7
        
        # If place name is mentioned in query, give high score
        if keywords["title"] in query_lower:
            score += 1.0
        
        # If address keywords are mentioned in query, add score
        for addr_part in keywords["address"].split():
            if addr_part in query_lower and len(addr_part) > 3:  # Ignore too short words
                score += 0.3
        
        # Check if place is open (if query cares about this)
        if "open" in query_lower or "hours" in query_lower or "closed" in query_lower:
            if keywords["is_open"]:
                score += 0.5
            else:
                score -= 0.5  # If closed, reduce score
        
        # Intent-specific additional scoring
        if intent == "parking":
            # If price is mentioned
            if "price" in query_lower or "cost" in query_lower or "fee" in query_lower:
                # Lower price gets higher score, but ensure price info exists
                if keywords["price_2h"] > 0:
                    price_score = 1.0 - min(keywords["price_2h"] / 15.0, 1.0)  # Assume 15 EUR as reference max price
                    score += price_score * 0.5  # Price weight 0.5
            
            # If spots count is mentioned
            if "spots" in query_lower or "space" in query_lower or "capacity" in query_lower:
                if keywords["spots"] > 0:
                    spots_score = min(keywords["spots"] / 400.0, 1.0)  # Assume 400 spots as reference
                    score += spots_score * 0.3  # Spots count weight 0.3
            
            # If height restriction is mentioned
            if "height" in query_lower or "restriction" in query_lower or "limit" in query_lower:
                if keywords["height"] > 0:
                    height_score = min(keywords["height"] / 2.5, 1.0)  # Assume 2.5m as reference
                    score += height_score * 0.3  # Height weight 0.3
        
        elif intent == "sushi":
            # If rating is mentioned
            if "rating" in query_lower or "review" in query_lower or "score" in query_lower:
                if keywords["rating"] > 0:
                    rating_score = keywords["rating"] / 5.0  # 5-point scale
                    score += rating_score * 0.6  # Rating weight 0.6
            
            # If price level is mentioned
            if "price" in query_lower or "cheap" in query_lower or "expensive" in query_lower:
                if keywords["price_level"] > 0:
                    # Assume price level 1-3, 1 is cheapest
                    price_score = 1.0 - ((keywords["price_level"] - 1) / 2.0)
                    score += price_score * 0.4  # Price weight 0.4
            
            # If specific cuisine is mentioned
            if "cuisine" in query_lower or "food" in query_lower or "dish" in query_lower:
                for food_type in keywords["food_types"]:
                    if food_type in query_lower:
                        score += 0.7
                        break
            
            # Special dietary requirements
            if "vegan" in query_lower or "vegetarian" in query_lower:
                for food_type in keywords["food_types"]:
                    if "vegan" in food_type.lower() or "vegetarian" in food_type.lower():
                        score += 0.8
                        break
        
        return score
    
    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def vector_similarity_search(self, query: str, intent: str, 
                                context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform complex semantic query through vector similarity"""
        # Get query embedding vector
        query_embedding_response = openai.Embedding.create(
            input=query,
            model=EMBEDDING_MODEL
        )
        query_embedding = query_embedding_response["data"][0]["embedding"]
        
        results = []
        embeddings = self.place_embeddings.get(intent, {})
        
        # Calculate similarity between query and all places
        for title, embedding in embeddings.items():
            similarity = 1 - spatial.distance.cosine(query_embedding, embedding)
            place_info = self.place_index[intent][title]
            
            results.append({
                "title": title,
                "data": place_info["data"],
                "similarity": similarity
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]  # Return top 5 results
    
    def retrieve_information(self, query: str, intent_info: Dict[str, Any], 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Choose the most appropriate retrieval strategy based on query type and return results"""
        intent = intent_info["intent"]
        if intent not in ["parking", "sushi"]:
            return {"error": "Unsupported intent type"}
        
        # Simple queries use filtering and sorting
        intent_processor = IntentProcessor()
        if intent_processor.is_simple_query(query, intent_info):
            results = self.direct_filter_search(query, intent, context)
            retrieval_method = "direct_filter"
        else:
            # Complex queries use vector similarity
            results = self.vector_similarity_search(query, intent, context)
            retrieval_method = "vector_similarity"
        
        # Enrich result data
        enriched_results = self._enrich_results(results, intent)
        
        return {
            "intent": intent,
            "query_type": intent_info["query_type"],
            "retrieval_method": retrieval_method,
            "results": enriched_results
        }
    
    def _enrich_results(self, results: List[Dict[str, Any]], intent: str) -> List[Dict[str, Any]]:
        """Enrich retrieval results, add more understandable formatted information"""
        enriched_results = []
        
        for result in results:
            place_data = result["data"]
            place = place_data["place"]
            
            enriched_item = {
                "title": place["title"],
                "distance": f"{place_data['distance']:.1f} meters",
                "address": place["formattedAddress"],
                "status": "Open" if place.get("businessHours", {}).get("status") == "OPEN" else "Closed",
                "business_hours": place.get("businessHours", {}).get("formattedHours", ["No business hours provided"]),
                "raw_data": place_data  # Keep raw data for later use
            }
            
            # Add type-specific extra information
            if intent == "parking":
                # Add parking specific information
                parking_info = place.get("parking", {})
                commercial = place.get("commercial", {})
                
                # Spots information
                if "spotsNumber" in parking_info:
                    enriched_item["total_spots"] = parking_info["spotsNumber"]
                if "freeSpotsNumber" in parking_info:
                    enriched_item["free_spots"] = parking_info["freeSpotsNumber"]
                
                # Price information
                if "priceSummary" in commercial:
                    enriched_item["price_summary"] = commercial["priceSummary"].get("priceSummaryText", "No price information provided")
                
                # Dimension restrictions
                if "parkingDimensionRestriction" in parking_info:
                    dim = parking_info["parkingDimensionRestriction"]
                    restrictions = []
                    if "height" in dim:
                        restrictions.append(f"Height limit: {dim['height']} meters")
                    if "width" in dim:
                        restrictions.append(f"Width limit: {dim['width']} meters")
                    if restrictions:
                        enriched_item["dimension_restrictions"] = restrictions
                
                # Payment methods
                if "paymentMethods" in commercial:
                    enriched_item["payment_methods"] = commercial["paymentMethods"]
            
            elif intent == "sushi":
                # Add restaurant specific information
                
                # Rating information
                if "reviewSummary" in place:
                    review = place["reviewSummary"]
                    enriched_item["rating"] = f"{review.get('averageRating', 0)}/5 ({review.get('reviewCount', 0)} reviews)"
                
                # Price information
                if "commercial" in place and "priceSummary" in place["commercial"]:
                    price_summary = place["commercial"]["priceSummary"]
                    if "priceRangeText" in price_summary:
                        enriched_item["price_level"] = price_summary["priceRangeText"]
                
                # Cuisine information
                if "foodTypes" in place:
                    enriched_item["cuisine"] = []
                    primary_cuisine = ""
                    for food in place["foodTypes"]:
                        if food.get("primary", False):
                            primary_cuisine = food["name"]
                        enriched_item["cuisine"].append(food["name"])
                    if primary_cuisine:
                        enriched_item["primary_cuisine"] = primary_cuisine
                
                # Contact information
                if "contact" in place:
                    contact = place["contact"]
                    enriched_item["contact"] = {}
                    if "phoneNumber" in contact:
                        enriched_item["contact"]["phone"] = contact["phoneNumber"]
                    if "website" in contact:
                        enriched_item["contact"]["website"] = contact["website"]
            
            enriched_results.append(enriched_item)
        
        return enriched_results

#########################################
# 4. Response Generation
#########################################

class ResponseGenerator:
    def __init__(self):
        """Initialize response generator"""
        pass
    
    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def generate_response(self, query: str, retrieved_data: Dict[str, Any], 
                         conversation_history: List[Dict[str, str]]) -> str:
        """Generate natural language response based on retrieved information"""
        system_prompt = """You are a professional in-car assistant helping users find and learn about sushi restaurants and parking facilities around Marienplatz in Munich.
Generate natural, helpful responses based on the provided data. Responses should be concise but informative, suitable for voice playback in a driving environment.
Use a polite, friendly tone but don't be overly verbose. If you can't find the information the user is asking about in the data, honestly say so."""
        
        # Combine conversation history and retrieval results into messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})
        messages.append({
            "role": "function", 
            "name": "retrieve_information", 
            "content": json.dumps(retrieved_data, ensure_ascii=False)
        })
        
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content

#########################################
# 5. Conversation Management
#########################################

class ConversationManager:
    def __init__(self, data_processor: DataProcessor):
        """Initialize conversation manager"""
        self.history = []
        self.current_context = {"intent": None, "selected_place": None}
        self.intent_processor = IntentProcessor()
        
        # Initialize data and retriever
        place_index, place_embeddings = data_processor.preprocess_all()
        self.retriever = InformationRetriever(place_index, place_embeddings)
        self.response_generator = ResponseGenerator()
    
    def process_query(self, user_query: str) -> str:
        """Process user query and generate response"""
        # 1. Intent recognition
        intent_info = self.intent_processor.detect_intent(user_query)
        
        # If user is discussing a specific place, maintain context
        if self.current_context["intent"] and intent_info["intent"] == "unknown":
            intent_info["intent"] = self.current_context["intent"]
            print(f"Maintaining context intent: {intent_info['intent']}")
        
        # 2. Information retrieval
        retrieved_data = self.retriever.retrieve_information(
            user_query, intent_info, self.current_context
        )
        
        # 3. Generate response
        response = self.response_generator.generate_response(
            user_query, retrieved_data, self.history
        )
        
        # 4. Update context and history
        self._update_context(user_query, response, retrieved_data)
        self.history.extend([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response}
        ])
        
        # If history is too long, keep only recent conversations
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        return response
    
    def _update_context(self, query: str, response: str, retrieved_data: Dict[str, Any]) -> None:
        """Update current conversation context"""
        self.current_context["intent"] = retrieved_data.get("intent", self.current_context["intent"])
        
        # Try to extract selected place from query or response
        if "results" in retrieved_data and retrieved_data["results"]:
            # Default first result as current focus
            self.current_context["selected_place"] = retrieved_data["results"][0]
            
            # Check if a specific place is mentioned
            for place in retrieved_data["results"]:
                title = place["title"].lower()
                if title in query.lower() or title in response.lower():
                    self.current_context["selected_place"] = place
                    break
    
    def display_conversation(self):
        """Display current conversation history, for Jupyter notebook demo"""
        for message in self.history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                display(Markdown(f"**User**: {content}"))
            elif role == "assistant":
                display(Markdown(f"**Assistant**: {content}"))
            else:
                display(Markdown(f"**{role}**: {content}"))
    
    def reset_conversation(self):
        """Reset conversation history and context"""
        self.history = []
        self.current_context = {"intent": None, "selected_place": None}
        return "Conversation has been reset."

#########################################
# Main Program
#########################################

class PlaceAssistant:
    def __init__(self):
        """Initialize place assistant"""
        print("Initializing data...")
        self.data_processor = DataProcessor()
        self.conversation_manager = ConversationManager(self.data_processor)
        print("Initialization complete! Ready to receive queries.")
    
    def query(self, user_input: str) -> str:
        """Process user input and return response"""
        if user_input.lower() in ["reset", "clear"]:
            return self.conversation_manager.reset_conversation()
        
        response = self.conversation_manager.process_query(user_input)
        return response
    
    def display_conversation(self):
        """Display conversation history"""
        self.conversation_manager.display_conversation()

# Demo execution example
def run_demo():
    """Run demo program"""
    assistant = PlaceAssistant()
    
    while True:
        user_input = input("\nEnter your query (type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = assistant.query(user_input)
        print(f"\nAssistant: {response}")
        print("\n" + "-"*50)

if __name__ == "__main__":
    run_demo()