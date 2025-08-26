# VBeing_Research システムアーキテクチャ クラス図

## 概要

このドキュメントはVBeing_Researchシステムアーキテクチャのクラス図を提供します。システムは主に関数型プログラミングアプローチを使用して実装されており、クラスベースの構造ではないため、これらの図は従来のクラス階層ではなく、コンポーネントとその関係を表しています。

## システムコンポーネント図

```mermaid
classDiagram
    class Webインターフェース {
        +Flaskアプリ
        +index()
        +initialize()
        +set_mode()
        +set_model()
        +send_message()
        +reset_conversation()
        +get_quota_usage()
        +serve_audio()
        +manage_system_prompts()
    }
    
    class コア処理エンジン {
        +run_gemini_command()
        +load_system_prompt()
        +get_gemini_response()
        +process_gemini_response()
        +query_with_retries()
        +select_optimal_model()
        +log_conversation()
        +log_token_usage()
    }
    
    class 音声データ管理 {
        +load_voice_data()
        +create_voice_data_json()
        +play_audio()
        +update_voice_data_csv()
        +get_next_audio_number()
        +get_available_clone_voices()
    }
    
    class 音声選択エンジン {
        +get_audio_selection_from_gemini()
        +find_best_match_text()
        +rag_audio_selection()
        +get_best_matching_audio()
        +detect_emotions()
        +categorize_voice_data()
        +calculate_context_similarity()
    }
    
    class 音声認識 {
        +listen_for_speech()
    }
    
    class 外部API {
        +Gemini API
        +Zonos API
    }
    
    Webインターフェース --> コア処理エンジン : 使用
    Webインターフェース --> 音声データ管理 : 使用
    Webインターフェース --> 音声認識 : 使用
    コア処理エンジン --> 外部API : インターフェース
    コア処理エンジン --> 音声選択エンジン : 使用
    コア処理エンジン --> 音声データ管理 : 使用
    音声選択エンジン --> 音声データ管理 : 使用
```

## 日本語モードでのZonos統合

```mermaid
classDiagram
    class Webインターフェース {
        +check_zonos_connection()
        +direct_tts()
        +generate_and_play_voice()
    }
    
    class Zonos音声生成 {
        +preprocess_text_for_zonos()
        +generate_zonos_voice_data()
        +save_zonos_voice_data()
        +generate_zonos_voice()
        +test_zonos_connection()
        +validate_zonos_text()
        +detect_language_from_system_prompt()
    }
    
    class ZonosAPI {
        +APIキー
        +モデル
        +音声生成
    }
    
    class システムプロンプト管理 {
        +load_mode_specific_system_prompt()
        +get_current_system_prompt()
        +save_current_system_prompt()
        +get_system_prompt_templates()
    }
    
    Webインターフェース --> Zonos音声生成 : 使用
    Zonos音声生成 --> ZonosAPI : インターフェース
    Webインターフェース --> システムプロンプト管理 : モード9で使用
    Zonos音声生成 --> システムプロンプト管理 : 言語検出に使用
```

## Zonos音声生成のデータフロー

```mermaid
sequenceDiagram
    participant ユーザー
    participant Webインターフェース
    participant コア処理エンジン
    participant Zonos音声生成
    participant ZonosAPI
    
    ユーザー->>Webインターフェース: テキスト入力（日本語）
    Webインターフェース->>コア処理エンジン: 入力処理
    コア処理エンジン->>Webインターフェース: 応答生成
    Webインターフェース->>Zonos音声生成: 応答の音声生成
    Zonos音声生成->>Zonos音声生成: テキスト前処理
    Zonos音声生成->>Zonos音声生成: テキスト検証
    Zonos音声生成->>Zonos音声生成: 言語検出
    Zonos音声生成->>ZonosAPI: テキストでリクエスト送信
    ZonosAPI->>Zonos音声生成: 音声データ返却
    Zonos音声生成->>Zonos音声生成: 音声ファイル保存
    Zonos音声生成->>Webインターフェース: 音声ファイル名返却
    Webインターフェース->>ユーザー: 音声応答再生
```

## コンポーネント関係

システムは関数型プログラミングアプローチを使用し、以下の主要な関係があります：

1. **Webインターフェース** (`web_interface.py`)
   - コア処理エンジンから関数をインポートして使用
   - HTTPルートとユーザー対話を処理
   - Zonosモード（モード9）を含むモード選択を管理

2. **コア処理エンジン** (`play_voice_with_gemini.py`)
   - ユーザー入力を処理するメインロジックを含む
   - 外部API（GeminiとZonos）とのインターフェース
   - 音声選択と生成を管理

3. **Zonos統合**
   - 両方のファイルの関数が連携して以下を実行：
     - 接続状態の確認
     - テキストの前処理と検証
     - Zonos APIを使用した音声生成
     - 音声ファイルの保存と提供

4. **システムプロンプト管理**
   - システムプロンプトの読み込みと保存を処理
   - 日本語Zonosテンプレートを含むモード固有のテンプレートを使用