from anthropic import Anthropic
from config import IDENTITY, TOOLS, MODEL, get_quote
from dotenv import load_dotenv

load_dotenv()

class ChatBot:
    def __init__(self, session_state):
        self.anthropic = Anthropic()
        self.session_state = session_state

    def generate_message(
        self,
        messages,
        max_tokens,
    ):
        try:
            message_list = []
            for msg in messages:
                if msg["role"] == "user":
                    message_list.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    message_list.append({"role": "assistant", "content": msg["content"]})

            response = self.anthropic.beta.messages.create(
                model=MODEL,
                system=IDENTITY,
                max_tokens=max_tokens,
                messages=message_list,
                tools=TOOLS,
            )
            return response
        except Exception as e:
            return {"error": str(e)}

    def process_user_input(self, user_input):
        self.session_state.messages.append({"role": "user", "content": user_input})

        response_message = self.generate_message(
            messages=self.session_state.messages,
            max_tokens=2048,
        )

        if "error" in response_message:
            return f"An error occurred: {response_message['error']}"

        try:
            content = response_message.content
            if len(content) > 0 and hasattr(content[-1], 'type'):
                if content[-1].type == "tool_calls":
                    tool_use = content[-1]
                    func_name = tool_use.tool_calls[0].function.name
                    func_params = tool_use.tool_calls[0].function.arguments
                    tool_call_id = tool_use.tool_calls[0].id

                    result = self.handle_tool_use(func_name, func_params)
                    self.session_state.messages.append(
                        {"role": "assistant", "content": content}
                    )
                    self.session_state.messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call_id,
                    })

                    follow_up_response = self.generate_message(
                        messages=self.session_state.messages,
                        max_tokens=2048,
                    )

                    if "error" in follow_up_response:
                        return f"An error occurred: {follow_up_response['error']}"

                    response_text = follow_up_response.content[0].text
                    self.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
                    return response_text
                
            response_text = content[0].text
            self.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
            return response_text
            
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def handle_tool_use(self, func_name, func_params):
        if func_name == "get_quote":
            premium = get_quote(**func_params)
            return f"Quote generated: ${premium:.2f} per month"
        
        raise Exception("An unexpected tool was used")