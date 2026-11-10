from client import client

schema = """
CREATE TABLE customers (id INT, name TEXT, country TEXT);
CREATE TABLE orders (id INT, customer_id INT, amount REAL, order_date TEXT);
CREATE TABLE products (id INT, name TEXT, price REAL);
CREATE TABLE order_items (order_id INT, product_id INT, quantity INT);
"""

question = "2024년에 가장 많이 산 고객 이름과 총액을 알려 줘"

response = client.chat.complete(
    model="codestral-latest",
    messages=[
        {
            "role": "system",
            "content": (
                "아래 스키마의 PostgreSQL 데이터베이스를 자연어 질문으로 조회하는 SQL만 작성해. "
                "SQL 코드 블록 외에는 설명을 붙이지 마.\n\n" + schema
            ),
        },
        {"role": "user", "content": question},
    ],
)
print(response.choices[0].message.content)
