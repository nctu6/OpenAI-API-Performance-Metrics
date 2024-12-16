import requests
import json
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Suppress only the single InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

def process_stream_line(line):
    try:
        # Decode the line from bytes to string if necessary
        if isinstance(line, bytes):
            line = line.decode('utf-8')
            
        # Remove the "data: " prefix if it exists
        if line.startswith('data: '):
            line = line[6:]
            
        # Parse the JSON content
        if line.strip() == '[DONE]':
            return None
            
        data = json.loads(line)
        
        # Extract the content from the response structure
        if 'choices' in data and len(data['choices']) > 0:
            if 'delta' in data['choices'][0] and 'content' in data['choices'][0]['delta']:
                return data['choices'][0]['delta']['content']
            
        return None
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"Error processing line: {str(e)}")
        return None

def make_chat_request(prompt):
    base_url = "https://www.familymart.unieai.com:8088"
    api_url = f"{base_url}/api/chat/completions"
    api_key = "sk-36944186ace74c55bdd98a70229dfc83"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "MyCustomAgent",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        total_chars = 0
        print("Making request...")
        
        with requests.post(
            api_url,
            headers=headers,
            json=payload,
            verify=False,
            timeout=180,
            stream=True  # Enable streaming
        ) as response:
            
            print(f"Response status: {response.status_code}")
            
            if response.ok:
                print("\nStreaming response:")
                for line in response.iter_lines():
                    if line:
                        content = process_stream_line(line)
                        if content:
                            print(content, end='', flush=True)  # Print content in real-time
                            total_chars += len(content)
                
                print(f"\n\nTotal characters received: {total_chars}")
                return total_chars
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None

    except requests.exceptions.Timeout:
        print("Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {str(e)}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    return None

if __name__ == "__main__":
    # Example usage
    result = make_chat_request("我今天的工作有什麼")
    if result is not None:
        print(f"Request completed successfully with {result} characters")