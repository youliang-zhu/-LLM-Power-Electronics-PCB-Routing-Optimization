import requests

class DEEPSEEKAPI:
    def __init__(self):
        self.model_name = "deepseek-coder-v2:16b"
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def get_llm_response(self, prompt: str) -> str:
        """向Ollama发送请求并获取响应，针对deepseek-coder优化"""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            # deepseek-coder专用参数
            "temperature": 0.1, 
            "num_predict": 2048,  
        }
        
        try:
            response = requests.post(self.ollama_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"Ollama API调用出错: {e}")
            return ""