# AUTOMATED-TRPG-SCRIPT-WORKSHOP-SYSTEM

An automated AI-powered scriptwriting workshop that generates full-length TRPG-style scripts. The system features a multi-agent framework where different AI models (Writer, Game Master, Actors, Director) collaborate and iterate on a story in real time.

這是一個創新的自動化劇本創作系統，它利用多個 AI 代理人（agent）協作，從零開始共同生成一個完整的 TRPG（桌上角色扮演遊戲）風格劇本。專案的核心思想是模擬一個真實的劇本工作坊，讓不同的 AI 角色分工合作，彼此激發創意，最終產出連貫且充滿戲劇張力的故事。

## 核心功能

  - **多代理人框架**：模擬一個完整的創作團隊，包含**世界架構師**、**遊戲主持人**、**即興演員**、**劇本監督員**、**觀眾**和**AI 導演**。
  - **動態故事生成**：程式會從頭開始生成故事藍圖、角色設定和開場情境，每次運行都能獲得一個獨一無二的故事。
  - **即時回饋循環**：故事在多個回合中逐步發展。每個回合都包含「即興表演」、「劇本整理」、「觀眾回饋」和「導演決策」等階段，讓故事能夠自我迭代和優化。
  - **強大模型兼容性**：專案基於 Hugging Face 的 `transformers` 庫，可以輕鬆替換不同的語言模型，以獲得更好的生成效果。

## 運作原理

系統以一個封閉的、多輪迴的循環運作：

1.  **世界架構師**生成故事藍圖。
2.  **遊戲主持人**根據藍圖設定開場場景。
3.  **演員**們根據場景描述和角色設定進行即興發揮。
4.  **劇本監督員**將演員的回應整理成劇本。
5.  **觀眾**對新生成的劇本提供回饋。
6.  **AI 導演**根據觀眾回饋做出宏觀決策，並指導遊戲主持人推進下一個回合的劇情。

這個循環重複執行，直到達到設定的回合數，最終將所有場景合併成一個完整的劇本。

-----
