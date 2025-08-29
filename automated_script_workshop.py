from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import random
from datetime import datetime
import threading
import queue
import re

class SimpleLLM:
    def __init__(self, model_name="microsoft/phi-2"):
        print(f"Loading model: {model_name}")
        start_time = time.time()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto", 
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded successfully. Time: {time.time() - start_time:.2f} seconds")
    
    def generate(self, prompt, max_tokens=300, temperature=0.7, stop_sequences=None):
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.model.device)
            
            # 設定生成參數
            generation_config = {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'do_sample': True,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            
            # 如果有停止序列，添加到生成配置
            if stop_sequences:
                generation_config['stopping_criteria'] = stop_sequences
            
            output = self.model.generate(
                **inputs,
                **generation_config
            )
            
            input_length = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(
                output[0][input_length:], 
                skip_special_tokens=True
            ).strip()
            
            return response
        except Exception as e:
            return f"Generation error: {str(e)}"

class WorkshopAgent:
    def __init__(self, name, role, system_prompt, llm):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm
        self.conversation_history = []
    
    def clean_response(self, response):
        """清理模型響應，移除無關內容"""
        if not response:
            return "[No response generated]"
            
        # 移除常見的無關模式
        patterns = [
            r'Exercise:.*',
            r'Answer:.*', 
            r'Essay.*',
            r'<\|endofgeneration\|>.*',
            r'###.*',
            r'Note:.*',
            r'Please note.*',
            r'Remember:.*',
            r'Key points:.*',
            r'In summary.*'
        ]
        
        for pattern in patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除多餘的空行
        response = re.sub(r'\n\s*\n', '\n\n', response)
        
        return response.strip()
    
    def respond(self, input_text):
        # 加強系統提示約束
        context = f"""{self.system_prompt}

IMPORTANT INSTRUCTIONS:
- Stay strictly in character as {self.name}
- Respond only to the current situation described
- Do not break character or discuss acting techniques
- Maintain the established fictional world
- No exercises, essays, or analysis - only in-character responses

Recent Context:
"""
        
        # 添加對話歷史（限制長度）
        for exchange in self.conversation_history[-2:]:
            context += f"Previous: {exchange['input'][:100]}...\n"
            context += f"My Response: {exchange['response'][:100]}...\n\n"
        
        context += f"Current Situation: {input_text}\n\nMy Response as {self.name}:"
        
        # 生成響應
        response = self.llm.generate(
            context, 
            max_tokens=250,  # 減少token數量以避免截斷
            temperature=0.7 if self.role.startswith('actor') else 0.8
        )
        
        # 清理響應
        response = self.clean_response(response)
        
        # 如果響應太短或無效，使用fallback
        if len(response.split()) < 3:
            if self.role.startswith('actor'):
                response = f"[{self.name} is carefully considering the situation...]"
            else:
                response = f"[{self.name} is processing the information...]"
        
        self.conversation_history.append({
            'input': input_text,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response

class TRPGWorkshopSystem:
    def __init__(self):
        self.llm = SimpleLLM()
        self.agents = {}
        self.session_log = []
        self.current_round = 0
        
        # 系統提示詞
        self.prompts = {
            'world_architect': """You are a top-tier concept designer for film studios. Your specialty is creating compelling worldviews, opening events, and core characters for script co-creation workshops.

Design a complete blueprint following the "Pressure Cooker Scenario" philosophy: place conflicting characters in a mysterious, enclosed environment with unknown variables, and ignite dramatic chemistry with a triggering event.

Output format must include:
1. Project codename
2. High concept logline
3. Tone & style keywords
4. Detailed scene description with sensory details
5. Inciting incident
6. 3 core characters with Want/Need/Contradiction/Secret
7. Central dramatic question
8. Areas of deliberate ambiguity for actors to explore

Respond only with the blueprint content. No additional commentary.""",

            'mc': """You are an experienced PbtA (Powered by the Apocalypse) game master. You excel at narrative-first principles and creating dramatic stories.

STRICT RULES:
- Portray only the fictional world and events
- Never break character or discuss game mechanics
- No exercises, essays, or analysis
- Only describe what happens in the world

Core principles:
- Portray a real world with rich sensory details
- Treat character existence as real
- Let character actions drive the plot
- Always state potential consequences and ask questions
- Make MC moves at appropriate moments

You know all character secrets and can use them to create targeted dramatic situations.""",

            'actor': """You are a professional improvisation actor trained in method acting and G.O.T.E. theory.

STRICT RULES:
- Always think and respond in first person as your character
- Base all actions on your character's knowledge only
- Never break character or discuss acting techniques
- No exercises, essays, or analysis - only in-character responses

Character setup will be provided separately. Respond only as your character would in the given situation.""",

            'script_supervisor': """You are a professional script supervisor. Your job is to integrate independent actor responses into a flowing, coherent scene log in proper script format.

STRICT RULES:
- Maintain the established world and characters
- Use only the provided character responses
- Format as proper screenplay with actions and dialogue
- Preserve the established atmosphere and tone
- No analysis, commentary, or additional content

Input: Multiple character responses
Output: Coherent scene log in screenplay format

Format rules:
1. Arrange events in logical chronological order
2. Format as proper screenplay:
   - Action descriptions in present tense
   - Character names in CAPITALS
   - Dialogue below character names
   - Internal monologue in (parentheses)
3. Add connective tissue for smooth narrative flow
4. Remain objective - only record what happened""",

            'audience': """You are an experienced film industry analyst and focus group member. You provide immediate, honest reactions to script workshop transcripts.

Analyze each scene log for:
1. Emotional Highlight: Which moment was most engaging? Why?
2. Character Appeal: Which character impressed you most? What feelings did they evoke?
3. Pacing & Clarity: Any confusing or boring parts? How's the rhythm?
4. Genre Feel: What film genre does this feel like?
5. Anticipation: What are you most curious about next?

Be direct and honest. You're the audience, not the writer - react and analyze, don't suggest changes.""",

            'ai_director': """You are the master control AI director of an automated script workshop. You coordinate all AI sub-models through a strict "Director's Algorithm":

STRICT RULES:
- Make decisions based only on the provided material
- Focus on narrative development and character arcs
- No exercises, essays, or unrelated content

LOOP PHASE:
1. GATHER: Collect responses from actors
2. SYNTHESIZE: Send to script supervisor for scene log
3. ANALYZE: Get audience feedback, make creative decisions
4. DISTRIBUTE: Create broadcast package with scene log + new director instructions

Decision logic:
- IF audience feedback shows "slow pacing" → require MC to make a hard move
- IF audience shows "confusion" → require MC to reveal key information  
- IF audience highly interested in character → focus narrative on them
- IF scene ends with strong suspense → require direct actor responses

Output format: Director's Notes with analysis, decision, and exact instructions for next round."""
        }
        
    def initialize_agents(self):
        """初始化所有工作坊代理"""
        agent_configs = [
            ('world_architect', 'World Designer', self.prompts['world_architect']),
            ('mc', 'Game Master', self.prompts['mc']), 
            ('script_supervisor', 'Script Supervisor', self.prompts['script_supervisor']),
            ('audience', 'Focus Group', self.prompts['audience']),
            ('ai_director', 'AI Director', self.prompts['ai_director'])
        ]
        
        for agent_id, name, prompt in agent_configs:
            self.agents[agent_id] = WorkshopAgent(name, agent_id, prompt, self.llm)
            print(f"Initialized: {name}")
    
    def validate_response(self, response, agent_type):
        """驗證響應是否在情境中"""
        if not response or len(response.strip()) < 10:
            return False
            
        response_lower = response.lower()
        
        # 檢查無效模式
        invalid_patterns = {
            'actor': [r'exercise', r'essay', r'answer:', r'analysis', r'theory', r'technique'],
            'mc': [r'rewrite', r'exercise', r'technique', r'mechanic'],
            'all': [r'<\|endofgeneration\|>', r'###', r'note:', r'please note']
        }
        
        patterns = invalid_patterns.get(agent_type, []) + invalid_patterns['all']
        for pattern in patterns:
            if re.search(pattern, response_lower):
                return False
        
        return True
    
    def get_valid_response(self, agent, prompt, max_retries=2):
        """獲取有效的響應，最多重試指定次數"""
        for attempt in range(max_retries):
            response = agent.respond(prompt)
            if self.validate_response(response, agent.role):
                return response
            print(f"Invalid response from {agent.name}, retrying... (attempt {attempt + 1})")
        
        # 如果所有重試都失敗，返回fallback響應
        if agent.role.startswith('actor'):
            return f"[{agent.name} takes a moment to assess the situation before acting...]"
        else:
            return f"[{agent.name} processes the information and prepares to continue...]"
    
    def create_actor(self, character_name, character_setup):
        """創建個別演員代理"""
        actor_prompt = f"""{self.prompts['actor']}

CHARACTER SETUP:
Name: {character_setup['name']}
Core Drive: {character_setup['want']}
Internal Need: {character_setup['need']}
Core Contradiction: {character_setup['contradiction']}
Secret: {character_setup['secret']}
Speaking Style: {character_setup.get('voice', 'Direct and authentic')}

You embody this character completely. All responses must reflect their personality, motivations, and current emotional state. Never break character."""
        
        agent_id = f"actor_{character_name.lower()}"
        self.agents[agent_id] = WorkshopAgent(f"Actor: {character_name}", agent_id, actor_prompt, self.llm)
        return agent_id
    
    def generate_blueprint(self):
        """生成初始工作坊藍圖"""
        print("\n=== GENERATING WORKSHOP BLUEPRINT ===")
        blueprint_request = "Please create a complete initial blueprint for a script co-creation workshop following the format and theories provided. Respond only with the blueprint content."
        
        blueprint = self.agents['world_architect'].respond(blueprint_request)
        print(f"\nBlueprint Generated:\n{blueprint}")
        return blueprint
    
    def extract_characters_from_blueprint(self, blueprint):
        """從藍圖中提取角色信息"""
        print("Extracting character information from blueprint...")
        
        extraction_agent = WorkshopAgent(
            "Character Extractor", 
            "extractor", 
            self.prompts['world_architect'], 
            self.llm
        )
        
        comprehensive_query = f"""Based on this blueprint:

{blueprint}

Please extract ALL character information in this EXACT format:

CHARACTER_COUNT: [number]

CHARACTER_1:
NAME: [character name]
DESCRIPTION: [one line description]
WANT: [external goal]
NEED: [internal growth needed]
CONTRADICTION: [personality contradiction]
SECRET: [hidden information]
VOICE: [speaking style]

CHARACTER_2:
NAME: [character name]
DESCRIPTION: [one line description]
WANT: [external goal]
NEED: [internal growth needed]
CONTRADICTION: [personality contradiction]
SECRET: [hidden information]
VOICE: [speaking style]

[Continue for all characters...]

Respond ONLY with the above format. Do not add any other text."""

        response = extraction_agent.respond(comprehensive_query)
        print(f"Character extraction response:\n{response}")
        
        characters = self.parse_character_extraction(response)
        
        if not characters:
            print("Failed to extract characters. Using fallback method...")
            characters = self.fallback_character_extraction(blueprint)
        
        return characters
    
    def parse_character_extraction(self, response):
        """解析結構化的角色提取響應"""
        characters = []
        lines = response.split('\n')
        current_char = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('CHARACTER_') and ':' in line:
                if current_char and 'name' in current_char:
                    characters.append(current_char)
                current_char = {}
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                if key == 'NAME':
                    current_char['name'] = value
                elif key == 'DESCRIPTION':
                    current_char['description'] = value
                elif key == 'WANT':
                    current_char['want'] = value
                elif key == 'NEED':
                    current_char['need'] = value
                elif key == 'CONTRADICTION':
                    current_char['contradiction'] = value
                elif key == 'SECRET':
                    current_char['secret'] = value
                elif key == 'VOICE':
                    current_char['voice'] = value
        
        if current_char and 'name' in current_char:
            characters.append(current_char)
        
        return characters
    
    def fallback_character_extraction(self, blueprint):
        """備用方法從藍圖文本中提取角色"""
        characters = []
        
        import re
        
        char_patterns = [
            r'Character \d+: (\w+)',
            r'(\w+) - (.+?)(?=Character|\n|$)',
            r'\d+\.\s*(\w+)\s*[-:]?\s*(.+)'
        ]
        
        for pattern in char_patterns:
            matches = re.findall(pattern, blueprint, re.MULTILINE | re.IGNORECASE)
            if matches:
                for i, match in enumerate(matches[:3]):
                    if isinstance(match, tuple):
                        name = match[0]
                        description = match[1].strip() if len(match) > 1 else f"Character {i+1}"
                    else:
                        name = match
                        description = f"Character {i+1}"
                    
                    characters.append({
                        'name': name,
                        'description': description,
                        'want': f"Achieve their primary objective",
                        'need': f"Overcome their internal conflict",
                        'contradiction': f"Conflicted between duty and desire",
                        'secret': f"Hiding a crucial truth",
                        'voice': f"Direct and determined"
                    })
                break
        
        if not characters:
            default_chars = ['Alex', 'Jordan', 'Morgan']
            for i, name in enumerate(default_chars):
                characters.append({
                    'name': name,
                    'description': f"Protagonist {i+1}",
                    'want': f"Solve the central mystery",
                    'need': f"Find inner peace",
                    'contradiction': f"Brave but haunted by past",
                    'secret': f"Connected to the main conflict",
                    'voice': f"Thoughtful and measured"
                })
        
        return characters
    
    def setup_characters_from_blueprint(self, blueprint):
        """從藍圖中設置角色"""
        print("\nExtracting character information from blueprint...")
        
        characters = self.extract_characters_from_blueprint(blueprint)
        
        if not characters:
            print("Warning: Could not extract characters from blueprint. Workshop cannot proceed.")
            return []
        
        actor_ids = []
        for char in characters:
            actor_id = self.create_actor(char['name'], char)
            actor_ids.append(actor_id)
            print(f"Created actor: {char['name']}")
        
        return actor_ids
    
    def roll_dice(self):
        """為PbtA系統擲2d6"""
        return random.randint(1, 6) + random.randint(1, 6)
    
    def run_automated_workshop(self, num_rounds=3):
        """運行完整的自動化工作坊"""
        print("\n=== STARTING AUTOMATED TRPG WORKSHOP ===")
        
        try:
            blueprint = self.generate_blueprint()
            actor_ids = self.setup_characters_from_blueprint(blueprint)
            
            if not actor_ids:
                print("ERROR: Failed to create any actors. Cannot proceed.")
                return
            
            print(f"Successfully created {len(actor_ids)} actors")
            
            mc_init = f"""WORKSHOP INITIALIZATION - STAY IN WORLD

You are the Game Master for this specific scenario:

{blueprint}

YOUR ROLE: Describe only what happens in the world. Do not break character.
Do not discuss acting techniques. Do not give exercises.

OPENING SCENE REQUIREMENTS:
- Describe the high-pressure cooker environment with sensory details
- Establish the three characters in this situation  
- End with a clear dramatic question that prompts character action
- Maintain the suspenseful, anxious tone

Begin the scene description:"""
            
            opening_scene = self.get_valid_response(self.agents['mc'], mc_init)
            print(f"\n=== OPENING SCENE ===\n{opening_scene}")
            
            current_broadcast = f"""OPENING SCENE:

{opening_scene}

This is the start of your adventure. Based on the scene described above, what is your character's immediate reaction and first action?"""
            
            for round_num in range(1, num_rounds + 1):
                print(f"\n=== ROUND {round_num} ===")
                
                try:
                    actor_responses = {}
                    for actor_id in actor_ids:
                        try:
                            response = self.get_valid_response(self.agents[actor_id], current_broadcast)
                            actor_responses[actor_id] = response
                            actor_name = self.agents[actor_id].name.split(': ')[-1]
                            print(f"\n{actor_name}: {response[:200]}...")
                        except Exception as e:
                            print(f"Error getting response from {actor_id}: {e}")
                            continue
                    
                    if not actor_responses:
                        print("No valid actor responses received. Ending workshop.")
                        break
                    
                    scene_synthesis = "\n\n".join([
                        f"[{self.agents[aid].name.split(': ')[-1]}]: {resp}" 
                        for aid, resp in actor_responses.items()
                    ])
                    
                    scene_log_request = f"""Please create a coherent scene log from these character responses:

{scene_synthesis}

Format this as a flowing narrative scene with proper character actions and dialogue. Stay true to the original blueprint setting."""
                    
                    scene_log = self.get_valid_response(self.agents['script_supervisor'], scene_log_request)
                    print(f"\n=== SCENE LOG ROUND {round_num} ===\n{scene_log[:500]}...")
                    
                    audience_feedback = self.get_valid_response(
                        self.agents['audience'], 
                        f"Scene Log #{round_num}:\n{scene_log}"
                    )
                    print(f"\n=== AUDIENCE FEEDBACK ===\n{audience_feedback[:300]}...")
                    
                    director_input = f"""DIRECTOR'S NOTES - Round #{round_num + 1} Activation

Latest Intelligence:

[1. Latest Scene Log]
{scene_log}

[2. Latest Audience Feedback] 
{audience_feedback}

Analyze this intelligence and provide your next Director's Notes with clear analysis, decisions, and broadcast package for the next round."""
                    
                    director_notes = self.get_valid_response(self.agents['ai_director'], director_input)
                    print(f"\n=== DIRECTOR'S NOTES ===\n{director_notes[:300]}...")
                    
                    next_broadcast = f"""BROADCAST PACKAGE - Round {round_num + 1}

Previous Scene Summary:
{scene_log[:500]}...

Director's Analysis:
{director_notes[:300]}...

Continue the story based on the above developments. What happens next?"""
                    
                    mc_response = self.get_valid_response(self.agents['mc'], next_broadcast)
                    print(f"\n=== MC RESPONSE ===\n{mc_response[:300]}...")
                    
                    current_broadcast = f"""CONTINUING STORY:

Latest Scene:
{scene_log[:300]}...

New Development:
{mc_response[:300]}...

Based on these latest events, what is your character's reaction and next action?"""
                    
                    self.session_log.append({
                        'round': round_num,
                        'actor_responses': {aid: resp[:200] for aid, resp in actor_responses.items()},
                        'scene_log': scene_log[:500],
                        'audience_feedback': audience_feedback[:200],
                        'director_notes': director_notes[:200],
                        'mc_response': mc_response[:200],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    print(f"\n--- Round {round_num} Complete ---")
                    
                except Exception as round_error:
                    print(f"Error in round {round_num}: {round_error}")
                    continue
            
            print("\n=== WORKSHOP COMPLETE ===")
            
        except Exception as e:
            print(f"Workshop error: {e}")
            import traceback
            traceback.print_exc()
    
    def save_session_log(self, filename=None):
        """保存完整工作坊會話到文件"""
        if filename is None:
            filename = f"trpg_workshop_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.session_log, f, ensure_ascii=False, indent=2)
        
        print(f"Session saved to {filename}")
    
    def generate_final_script(self):
        """將所有場景日誌編譯成最終劇本格式"""
        final_script = "# TRPG Workshop Generated Script\n\n"
        
        for round_data in self.session_log:
            final_script += f"## Scene {round_data['round']}\n\n"
            final_script += round_data['scene_log'] + "\n\n"
            final_script += "---\n\n"
        
        return final_script

def main():
    print("=== AUTOMATED TRPG SCRIPT WORKSHOP SYSTEM ===")
    print("Initializing AI agents...")
    
    workshop = TRPGWorkshopSystem()
    workshop.initialize_agents()
    
    print("\nAll agents initialized successfully!")
    print("Starting automated workshop...")
    
    try:
        workshop.run_automated_workshop(num_rounds=3)
        
        final_script = workshop.generate_final_script()
        print(f"\n=== FINAL GENERATED SCRIPT ===\n{final_script}")
        
        workshop.save_session_log()
        
    except KeyboardInterrupt:
        print("\nWorkshop interrupted by user")
    except Exception as e:
        print(f"Workshop error: {str(e)}")
    
    print("\nWorkshop session complete!")

if __name__ == "__main__":
    main()
