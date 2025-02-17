import os
import json
import time
import threading
import re
from flask import Flask, request, render_template_string, redirect, url_for, flash, jsonify

try:
    from openai import OpenAI  # 環境に合わせて利用してください
except ImportError:
    # openaiパッケージが無い場合のダミー
    class OpenAI:
        def __init__(self, base_url=None, api_key=None):
            pass
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, timeout=15):
                    return type('dummy', (object,), {
                        "choices": [type('msg', (object,), {
                            "message": type('msg_content', (object,), {
                                "content": "LLMの応答(ダミー)"
                            })()
                        })]
                    })()

###############################################################################
# 1. 基本設定ファイル管理 (config.json)
###############################################################################

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "chat": {
        "model": "gemma2",
        "system": "あなたは会話エージェントです。",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    },
    "conversation": {
        "MAX_CONVERSATION_LENGTH": 100,
        "MAX_DISPLAY_MESSAGES": 50,
        "CONTEXT_WINDOW": 10
    },
    # 余計な番号を出力しないように、行頭数字や「<名前>:」は禁止と指示
    "prompt_instructions": (
        "以下のルールで投稿してください。\n"
        "・短めの文体で、チャットでの会話の雰囲気を意識する。\n"
        "・必要があれば文中に「@相手の名前」と書いて返信対象を示す（返信は1人のみ）。\n"
        "・行頭に投稿番号や「<名前>:」の形式を入れないでください。\n"
        "・数字やコロンから始まる行は書かないでください。\n"
        "システム側で投稿番号を付けます。"
    ),
    "conversation_log_file": "conversation_log.jsonl"
}

config = {}

def load_config():
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")
            config = DEFAULT_CONFIG.copy()
    else:
        config = DEFAULT_CONFIG.copy()
        save_config()

def save_config():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"設定ファイルの保存エラー: {e}")

###############################################################################
# 2. スレッド固有の設定 (メモリ上)
###############################################################################
# スレッドタイトル・エージェント一覧・まとめテキストなどを管理し、
# エクスポート/インポートでコピー・貼り付けできる形にします。

thread_config = {
    "title": "未設定",
    "agents": [],  # [{id, name, personality}, ...]
    "summary": ""  # 現在のまとめ
}
next_agent_id = 1

def export_thread_config_json():
    """thread_configをJSON文字列にして返す"""
    return json.dumps(thread_config, ensure_ascii=False, indent=2)

def import_thread_config_json(json_str):
    """JSON文字列を読み取り、thread_configを上書きする"""
    global thread_config, next_agent_id
    data = json.loads(json_str)
    if "title" in data:
        thread_config["title"] = data["title"]
    if "agents" in data and isinstance(data["agents"], list):
        thread_config["agents"] = data["agents"]
        if thread_config["agents"]:
            next_agent_id = max(a["id"] for a in thread_config["agents"]) + 1
        else:
            next_agent_id = 1
    if "summary" in data:
        thread_config["summary"] = data["summary"]

###############################################################################
# 3. 会話ログ管理 (メモリ + 任意のファイル保存)
###############################################################################

CONVERSATION_FILE = "conversation.json"

conversation = []
post_counter = 1

def load_conversation():
    """サーバー起動時に会話履歴を復元"""
    global conversation, post_counter
    if os.path.exists(CONVERSATION_FILE):
        try:
            with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            conversation = data.get("messages", [])
            post_counter = (conversation[-1]["number"] + 1) if conversation else 1
        except Exception as e:
            print(f"会話履歴の読み込みエラー: {e}")
            conversation = []
            post_counter = 1
    else:
        conversation = []
        post_counter = 1

def save_conversation():
    """conversation.json に会話を保存"""
    data = {"messages": conversation}
    try:
        with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"会話履歴の保存エラー: {e}")

def log_message_to_file(message):
    """1行ずつJSONで追記保存するログファイル"""
    log_file = config.get("conversation_log_file", "conversation_log.jsonl")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"会話ログの保存エラー: {e}")

###############################################################################
# 4. Chatクラス (LLM呼び出し)
###############################################################################

class Chat:
    def __init__(self):
        load_config()  # 最新のconfigを毎回読み込む
        self.model = config["chat"].get("model", "gemma2")
        self.system = config["chat"].get("system", "あなたは会話エージェントです。")
        base_url = config["chat"].get("base_url", "http://localhost:11434/v1")
        api_key = config["chat"].get("api_key", "ollama")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, user_message, system_override=None):
        sys_msg = system_override or self.system
        msgs = [{"role": "system", "content": sys_msg}] if sys_msg else []
        msgs.append({"role": "user", "content": user_message})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                timeout=60
            )
            text = response.choices[0].message.content
            return re.sub(r'<[^>]+>.*?</[^>]+>', '', text, flags=re.DOTALL) 
        except Exception as e:
            return f"エラー: {str(e)}"

chat_instance = Chat()

###############################################################################
# 5. 自動要約設定
###############################################################################
AUTO_SUMMARY_INTERVAL = 10  # 10件ごとにまとめ自動更新
messages_since_last_summary = 0

def generate_summary():
    """直近の会話をLLMで要約し、thread_config["summary"]に反映"""
    global thread_config
    # 直近50件程度を対象に
    recent_msgs = conversation[-AUTO_SUMMARY_INTERVAL:]
    conversation_text = ""
    for msg in recent_msgs:
        conversation_text += f"{msg['number']}. {msg['agent']}"
        if msg['reply_to']:
            conversation_text += f" (>>{msg['reply_to']})"
        conversation_text += f": {msg['text']}\n"

    prompt = (
        "以下はこれまでの会話の要約です。"
        + thread_config["summary"]
        + "\n以下は直近の会話です。これをこれまでの要約にマージし、論点や結論を整理してください。\n\n"
        + conversation_text
        + "以下はスレッドタイトルです。このタイトルに従って整理して下さい。"
        + thread_config["title"]
        + "\n[要約出力]:"
    )
    result = chat_instance(prompt, system_override="あなたは優秀な議論の要約者です。")
    thread_config["summary"] = result.strip()

def on_new_message_posted():
    """
    新しいメッセージが投稿された後に呼び出されるフック。
    - カウンタを進めてAUTO_SUMMARY_INTERVALに達したら自動で要約生成
    """
    global messages_since_last_summary
    messages_since_last_summary += 1
    if messages_since_last_summary >= AUTO_SUMMARY_INTERVAL:
        generate_summary()
        messages_since_last_summary = 0

###############################################################################
# 6. Flaskアプリ設定
###############################################################################

app = Flask(__name__)
app.secret_key = "secret_key_for_session"

load_config()
load_conversation()

###############################################################################
# 7. バックグラウンド会話ワーカー (エージェント投稿)
###############################################################################

def conversation_worker():
    global conversation, post_counter
    while True:
        # エージェントがいなければスキップ
        if not thread_config["agents"]:
            time.sleep(2)
            continue

        for agent in thread_config["agents"]:
            # configを再読込(例えばCONTEXT_WINDOWなどが変わったら即反映)
            load_config()
            chat = Chat()

            context_window = config["conversation"].get("CONTEXT_WINDOW", 10)
            prompt_instructions = config.get("prompt_instructions", "")
            max_len = config["conversation"].get("MAX_CONVERSATION_LENGTH", 100)

            # 直近の書き込み
            recent = conversation[-context_window:] if len(conversation) >= context_window else conversation

            # まとめがある場合も、エージェントが参照できるように追加
            summary_text = thread_config["summary"].strip()
            # プロンプト作成
            prompt = ""
            if thread_config["title"]:
                prompt += f"このスレッドのタイトルは「{thread_config['title']}」です。\n"
            if summary_text:
                prompt += f"現在のまとめ(要約)があります。参考にしてください:\n{summary_text}\n\n"
            prompt += f"あなたは {agent['name']}。性格は {agent['personality']}です。\n"
            if thread_config["title"]:
                prompt += "タイトルに配慮した発言を心がけてください。\n"
            prompt += "以下、直近の書き込みです。\n"
            for msg in recent:
                if msg.get("reply_to"):
                    prompt += f"{msg['number']}. {msg['agent']} (返信先: {msg['reply_to']}): {msg['text']}\n"
                else:
                    prompt += f"{msg['number']}. {msg['agent']}: {msg['text']}\n"
            prompt += "\n" + prompt_instructions

            response_text = chat(prompt).strip()
            if response_text:
                # 返信先を抽出
                valid_names = {a['name'] for a in thread_config["agents"]} | {m['agent'] for m in conversation}
                reply_to = None
                if valid_names:
                    pattern = r'@(' + '|'.join(re.escape(name) for name in valid_names) + r')'
                    match = re.search(pattern, response_text)
                    if match:
                        reply_to = match.group(1)
                        response_text = response_text.replace("@" + reply_to, "").strip()

                response_text = response_text.split('@')[0]
                new_msg = {
                    "number": post_counter,
                    "agent": agent["name"],
                    "reply_to": reply_to,
                    "text": response_text,
                    "timestamp": time.time()
                }
                conversation.append(new_msg)
                post_counter += 1
                log_message_to_file(new_msg)
                save_conversation()

                # 自動要約判定
                on_new_message_posted()

                # 長さ制限
                if len(conversation) > max_len:
                    conversation = conversation[-max_len:]
                    save_conversation()

            time.sleep(1)

        time.sleep(5)

threading.Thread(target=conversation_worker, daemon=True).start()

###############################################################################
# 8. テンプレート (ベース)
###############################################################################

BASE_HTML = '''
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>エージェント会話システム</title>
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
  <style>
    body {
      background-color: #f8f9fa;
      font-family: "メイリオ", sans-serif;
      margin-bottom: 50px;
    }
    .post {
      border-bottom: 1px solid #ddd;
      padding: 5px;
      margin-bottom: 10px;
    }
    .post-number {
      font-weight: bold;
      margin-right: 5px;
    }
    .post-agent {
      color: #007bff;
      font-weight: 600;
    }
    .post-reply {
      color: #28a745;
      margin-left: 5px;
    }
    .conversation {
      background-color: #ffffff;
      padding: 15px;
      border: 1px solid #dee2e6;
      max-height: 500px;
      overflow-y: auto;
    }
    .conversation-container {
      margin-bottom: 20px;
    }
    .agent-list-item {
      margin-bottom: 5px;
    }
  </style>
</head>
<body>
<div class="container mt-3">
  <h1 class="text-center mb-4">エージェント会話システム</h1>
  <div class="mb-3 text-right">
    <a href="{{ url_for('index') }}" class="btn btn-secondary">会話画面</a>
    <a href="{{ url_for('config_page') }}" class="btn btn-info">基本設定</a>
    <a href="{{ url_for('summary_page') }}" class="btn btn-warning">まとめページ</a>
  </div>
  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <div class="alert alert-warning">
        {% for msg in messages %}
          <div>{{ msg }}</div>
        {% endfor %}
      </div>
    {% endif %}
  {% endwith %}
  {{ body|safe }}
</div>
</body>
</html>
'''

###############################################################################
# 9. メイン画面 (会話表示 + ユーザ投稿 + エージェント管理)
###############################################################################

@app.route('/', methods=['GET'])
def index():
    max_display = config["conversation"].get("MAX_DISPLAY_MESSAGES", 50)
    display_conversation = conversation[-max_display:]

    # 初期描画用HTML
    conversation_html = render_template_string('''
    {% for msg in msgs|reverse %}
      <div class="post">
        <span class="post-number">{{ msg.number }}.</span>
        <span class="post-agent">{{ msg.agent }}</span>
        {% if msg.reply_to %}
          <span class="post-reply">>>{{ msg.reply_to }}</span>
        {% endif %}
        : {{ msg.text }}
      </div>
    {% endfor %}
    ''', msgs=display_conversation)

    body = render_template_string('''
      <div class="row">
        <!-- 左カラム: スレッド情報・会話 -->
        <div class="col-md-8">
          <h4>スレッドタイトル: {{ thread_config["title"] }}</h4>
          <form method="post" action="/set_thread_title" class="form-inline mb-3">
            <label class="mr-2">スレタイ入力:</label>
            <input type="text" name="thread_title" class="form-control mr-2" placeholder="スレッドタイトル">
            <button type="submit" class="btn btn-primary">設定</button>
          </form>

          <h4>会話スレッド (自動更新)</h4>
          <div class="conversation-container">
            <div class="conversation" id="conversation-container">
              {{ conversation_html|safe }}
            </div>
          </div>

          <form method="post" action="/clear_conversation" class="d-inline mb-3">
            <button type="submit" class="btn btn-warning">会話履歴をクリア</button>
          </form>
          <a href="/" class="btn btn-secondary mb-3">更新</a>

          <h5>あなたも参加</h5>
          <form method="post" action="/post_user_message" class="form-inline mb-3">
            <input type="text" name="username" class="form-control mr-2" placeholder="お名前" required>
            <input type="text" name="message" class="form-control mr-2" placeholder="メッセージ" required>
            <button type="submit" class="btn btn-success">送信</button>
          </form>
        </div>

        <!-- 右カラム: エージェント管理 & スレッド設定エクスポート/インポート -->
        <div class="col-md-4">
          <h4>エージェント管理</h4>
          <form method="post" action="/add_agent" class="mb-3">
            <div class="form-group">
              <label>名前:</label>
              <input type="text" name="name" class="form-control" required>
            </div>
            <div class="form-group">
              <label>性格:</label>
              <input type="text" name="personality" class="form-control" required>
            </div>
            <button type="submit" class="btn btn-primary btn-block">エージェント追加</button>
          </form>
          <h5>登録済みエージェント</h5>
          <ul class="list-group mb-4">
            {% for agent in thread_config["agents"] %}
              <li class="list-group-item d-flex justify-content-between align-items-center agent-list-item">
                <span>{{ agent.name }} ({{ agent.personality }})</span>
                <form method="post" action="/delete_agent/{{ agent.id }}" onsubmit="return confirm('このエージェントを削除してもよろしいですか？');">
                  <button type="submit" class="btn btn-danger btn-sm">削除</button>
                </form>
              </li>
            {% else %}
              <li class="list-group-item">エージェントが登録されていません。</li>
            {% endfor %}
          </ul>

          <h4>スレッド設定のエクスポート/インポート</h4>
          <p class="small">コピー&ペーストで設定を保存・復元できます。</p>
          <div class="mb-3">
            <button type="button" class="btn btn-info" onclick="exportThreadConfig()">エクスポート</button>
          </div>
          <form method="post" action="/import_thread_config">
            <div class="form-group">
              <textarea name="thread_config_json" id="threadConfigJson" class="form-control" rows="5"></textarea>
            </div>
            <button type="submit" class="btn btn-primary">インポート</button>
          </form>
        </div>
      </div>

      <!-- スレッド設定をAjaxで取得して表示する -->
      <script>
      function exportThreadConfig() {
        fetch('/export_thread_config')
          .then(response => response.json())
          .then(data => {
            document.getElementById('threadConfigJson').value = JSON.stringify(data, null, 2);
          })
          .catch(err => alert('エクスポート失敗: ' + err));
      }

      // 2秒おきに会話部分を更新
      function updateConversation() {
        fetch('/conversation_partial')
          .then(response => response.text())
          .then(html => {
            document.getElementById('conversation-container').innerHTML = html;
          })
          .catch(err => console.log("Error fetching conversation partial:", err));
      }
      setInterval(updateConversation, 2000);
      </script>
    ''', conversation_html=conversation_html, thread_config=thread_config)

    return render_template_string(BASE_HTML, body=body)

###############################################################################
# 10. 会話部分のみ返すエンドポイント (Ajax用)
###############################################################################

@app.route('/conversation_partial', methods=['GET'])
def conversation_partial():
    max_display = config["conversation"].get("MAX_DISPLAY_MESSAGES", 50)
    display_conversation = conversation[-max_display:]
    partial_html = render_template_string('''
    {% for msg in msgs|reverse %}
      <div class="post">
        <span class="post-number">{{ msg.number }}.</span>
        <span class="post-agent">{{ msg.agent }}</span>
        {% if msg.reply_to %}
          <span class="post-reply">>>{{ msg.reply_to }}</span>
        {% endif %}
        : {{ msg.text }}
      </div>
    {% endfor %}
    ''', msgs=display_conversation)
    return partial_html

###############################################################################
# 11. 投稿・エージェント管理
###############################################################################

@app.route('/post_user_message', methods=['POST'])
def post_user_message():
    global post_counter
    username = request.form.get("username", "").strip()
    message = request.form.get("message", "").strip()
    if username and message:
        new_msg = {
            "number": post_counter,
            "agent": username,
            "reply_to": None,
            "text": message,
            "timestamp": time.time()
        }
        conversation.append(new_msg)
        post_counter += 1
        log_message_to_file(new_msg)
        save_conversation()
        on_new_message_posted()

        max_len = config["conversation"].get("MAX_CONVERSATION_LENGTH", 100)
        if len(conversation) > max_len:
            conversation[:] = conversation[-max_len:]
            save_conversation()
    return redirect(url_for('index'))

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    global conversation, post_counter
    conversation = []
    post_counter = 1
    save_conversation()
    return redirect(url_for('index'))

@app.route('/add_agent', methods=['POST'])
def add_agent():
    global next_agent_id
    name = request.form.get("name", "").strip()
    personality = request.form.get("personality", "").strip()
    if name and personality:
        agent = {"id": next_agent_id, "name": name, "personality": personality}
        next_agent_id += 1
        thread_config["agents"].append(agent)
    return redirect(url_for('index'))

@app.route('/delete_agent/<int:agent_id>', methods=['POST'])
def delete_agent(agent_id):
    agents = thread_config["agents"]
    thread_config["agents"] = [a for a in agents if a["id"] != agent_id]
    return redirect(url_for('index'))

@app.route('/set_thread_title', methods=['POST'])
def set_thread_title():
    title = request.form.get("thread_title", "").strip()
    thread_config["title"] = title if title else "未設定"
    return redirect(url_for('index'))

@app.route('/export_thread_config', methods=['GET'])
def export_thread_config():
    return jsonify(thread_config)

@app.route('/import_thread_config', methods=['POST'])
def import_thread_config():
    global next_agent_id
    json_str = request.form.get("thread_config_json", "")
    try:
        import_thread_config_json(json_str)
        flash("スレッド設定をインポートしました。")
    except Exception as e:
        flash(f"インポート失敗: {e}")
    return redirect(url_for('index'))

###############################################################################
# 12. 基本設定ページ (config.json)
###############################################################################

@app.route('/config', methods=['GET', 'POST'])
def config_page():
    global config, chat_instance
    if request.method == "POST":
        config_text = request.form.get("config_text", "")
        try:
            new_config = json.loads(config_text)
            config = new_config
            save_config()
            chat_instance = Chat()
            flash("基本設定を更新しました。")
        except Exception as e:
            flash(f"設定更新エラー: {e}")
        return redirect(url_for('config_page'))
    else:
        body = render_template_string('''
          <h2>基本設定 (config.json)</h2>
          <p>LLMのAPI接続情報や会話ウィンドウサイズなど。</p>
          <form method="post">
            <div class="form-group">
              <textarea name="config_text" class="form-control" rows="20">{{ config_json }}</textarea>
            </div>
            <button type="submit" class="btn btn-primary">更新</button>
          </form>
          <a href="{{ url_for('index') }}" class="btn btn-secondary mt-3">会話画面へ戻る</a>
        ''', config_json=json.dumps(config, ensure_ascii=False, indent=2))
        return render_template_string(BASE_HTML, body=body)

###############################################################################
# 13. まとめページ
###############################################################################

@app.route('/summary', methods=['GET'])
def summary_page():
    body = render_template_string('''
      <h2>議論まとめ</h2>
      <p>現在の会話を要約したものを表示しています。一定数の投稿があると自動更新されますが、手動更新も可能です。</p>
      <div class="card mb-3">
        <div class="card-body" style="white-space: pre-wrap;">{{ thread_config["summary"] if thread_config["summary"] else "まだまとめはありません。" }}</div>
      </div>
      <form method="post" action="/generate_summary">
        <button type="submit" class="btn btn-warning">まとめを更新</button>
      </form>
      <a href="{{ url_for('index') }}" class="btn btn-secondary mt-3">会話画面へ戻る</a>
    ''', thread_config=thread_config)
    return render_template_string(BASE_HTML, body=body)

@app.route('/generate_summary', methods=['POST'])
def generate_summary_route():
    generate_summary()
    flash("まとめを更新しました。")
    return redirect(url_for('summary_page'))

###############################################################################
# 14. アプリ起動
###############################################################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678, debug=True)