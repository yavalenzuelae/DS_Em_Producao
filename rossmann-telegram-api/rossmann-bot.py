import requests
import json
import os
import pandas as pd
from flask import Flask, request, Response # para pegar mensagens pro bot no telegram

# constants
# token para interagir com o bot
TOKEN = '5922864156:AAGdjJMNvzQEVRWoVOrRrQLPDuJ7sRCFGyA'

# # info do bot
# https://api.telegram.org/bot5922864156:AAGdjJMNvzQEVRWoVOrRrQLPDuJ7sRCFGyA/getMe

# # receber atualizações do bot
# https://api.telegram.org/bot5922864156:AAGdjJMNvzQEVRWoVOrRrQLPDuJ7sRCFGyA/getUpdates # ou webhooks

# # receber atualizações do bot com webhooks
# https://api.telegram.org/bot5922864156:AAGdjJMNvzQEVRWoVOrRrQLPDuJ7sRCFGyA/setWebhook?url=https://ca1b7289e178bc.lhr.life

# # enviar mensagem
# https://api.telegram.org/bot5922864156:AAGdjJMNvzQEVRWoVOrRrQLPDuJ7sRCFGyA/sendMessage?chat_id=5951301168&text=Oi Yovanny. Estou bem.

# enviar mensagens pro bot no telegram
def send_message(chat_id, text):
    
    url = 'https://api.telegram.org/bot{}/'.format(TOKEN) # observar que o TOKEN fica como uma variável global
    url = url + '/sendMessage?chat_id={}'.format(chat_id) 
    
    # enviar mensagem pro bot
    r = requests.post(url,json={'text': text}) # isto é equivalente a copiar e colar a url no navegador na mão
    print('Código do status da requisição: {}'.format(r.status_code)) # verificar se deu certo a nossa requisição
    
    return None # porque somente estou enviando uma mensagem para esse usuário

def load_dataset(store_id):
    
    # carregar dados
    df10 = pd.read_csv('rossmann-telegram-api/test.csv')
    # df10 = pd.read_csv('test.csv')
    df_store_raw = pd.read_csv('rossmann-telegram-api/store.csv')
    # df_store_raw = pd.read_csv('store.csv')

    # juntar datasets de teste e lojas
    df_test = pd.merge(df10, df_store_raw, how='left', on='Store')

    # escolher lojas para predição
    df_test = df_test[df_test['Store'] == store_id]

    if not df_test.empty:
        # apagar datas com lojas fechadas
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[~df_test['Open'].isnull()]
        df_test = df_test.drop('Id', axis=1)

        # converter dataframe para json
        data = json.dumps(df_test.to_dict(orient='records'))
    else:
        data = 'error'   

    return data
        
def predict(data_teste):        
    # chamar API
    url = 'https://rossmann-api-j9wy.onrender.com/rossmann/predict'
    header = {'Content-type': 'application/json' }

    # status da requisição
    r = requests.post( url, data=data_teste, headers=header)
    print('Código do status da requisição: {}'.format(r.status_code))

    # converter json para dataframe
    d1 = pd.DataFrame(r.json(), columns=r.json()[0].keys())

    return d1

def parce_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']    
    store_id = store_id.replace('/', '') # remover a barra que tem toda mensagem do telegram    
    try:
        store_id = int(store_id)
    except ValueError:
        store_id = 'error'
    return chat_id, store_id

# Inicializar API
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        message = request.get_json()
        chat_id, store_id = parce_message(message)
       
        if store_id != 'error':
            # loading data
            data = load_dataset(store_id)
          
            if data != 'error':
                # prediction
                d1 = predict(data)
                # calculation
                d2 = d1[['store', 'prediction']].groupby('store').mean().reset_index()
                # send message
                msg = 'A loja {} vai realizar {:,.0f} vendas nas próximas 6 semanas'.format(d2['store'].values[0], d2['prediction'].values[0])
                send_message(chat_id, msg)
                return Response('Ok', status=200)
            else:
                send_message(chat_id, 'Loja não disponível')
                return Response('Ok', status=200)
        else:       
            send_message(chat_id, 'store_id errado')
            return Response('Ok', status=200) # se esquecer passar esse status a api fica rodando indefinidadamente pq acha que não terminou ... aí vc tem que mandar a resposta falando para ela que esté tudo bem
    else:
        return '<h1> Rossmann Telegram BOT </h1>'
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=port)
