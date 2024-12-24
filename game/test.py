import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 手动指定ChromeDriver的路径
driver_path = r"D:\chromedriver-win64\chromedriver.exe"

# 设置Selenium WebDriver
driver = webdriver.Chrome(executable_path=driver_path)

# 打开Pixiv的触手搜索页面
search_url = "https://www.pixiv.net/en/tags/触手"
driver.get(search_url)

# 等待页面加载完成
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "title")))

# 存储所有用户名和符合条件的作者名（使用 set 来去重）
all_authors = set()  # 存储所有抓取到的用户名
matching_authors = set()  # 存储符合条件的用户名（以“こ”开头）
visited_user_ids = set()  # 存储已经访问过的用户ID，避免重复抓取

# 翻页逻辑
while True:
    # 获取当前页面HTML内容
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # 查找所有作者的链接
    work_links = soup.find_all('a', href=True)

    # 打印所有抓取到的作者名
    print("当前页面的作者名：")
    for link in work_links:
        user_link = link.get('href')

        # 检查user_link是否包含作者的ID或用户名
        if 'user' in user_link:
            user_id = user_link.split('/')[-1]
            user_name = user_id  # 暂时假设user_name就是ID，实际情况可能不同

            # 如果该用户ID已经访问过，则跳过
            if user_id in visited_user_ids:
                print(f"跳过已访问的用户ID：{user_id}")
                continue  # 跳过当前循环，直接进入下一个用户

            # 记录已访问的用户ID
            visited_user_ids.add(user_id)

            print(f"抓取到的作者ID：{user_id}, 用户链接：{user_link}, 用户名：{user_name}")

            # 模拟点击进入用户页面
            user_page_url = "https://www.pixiv.net" + user_link  # 拼接出完整的用户页面URL
            driver.get(user_page_url)  # 进入该用户页面

            # 等待用户页面加载（尝试等待一个更加通用的元素）
            try:
                # 假设用户名在一个h1标签中，且类名为"user-name"（请根据实际情况调整）
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))

                # 获取用户页面HTML
                user_html = driver.page_source
                user_soup = BeautifulSoup(user_html, 'html.parser')

                # 从用户页面提取用户名
                username_element = user_soup.find('h1')  # 假设用户名位于<h1>标签中
                if username_element:
                    username = username_element.get_text().strip()
                    print(f"用户页面的用户名：{username}")

                    # 添加到所有用户名集合
                    all_authors.add(username)

                    # 检查用户名是否以“こ”开头
                    if username.startswith("こ"):
                        matching_authors.add(username)  # 使用 set 来去重

                # 保存到JSON文件（所有用户名和符合条件的用户名）
                with open('all_authors.json', 'w', encoding='utf-8') as f:
                    json.dump(list(all_authors), f, ensure_ascii=False, indent=4)
                print(f"所有作者名已保存到 'all_authors.json' 文件")

                with open('matching_authors.json', 'w', encoding='utf-8') as f:
                    json.dump(list(matching_authors), f, ensure_ascii=False, indent=4)
                print(f"符合条件的作者名已保存到 'matching_authors.json' 文件")

            except Exception as e:
                print(f"错误：无法提取用户名 {e}")
            finally:
                # 返回搜索页面
                driver.back()
                time.sleep(2)  # 等待页面加载

    # 输出符合条件的作者名
    if matching_authors:
        print(f"符合条件的作者名（以'こ'开头）：")
        for author in matching_authors:
            print(author)
    else:
        print("未找到符合条件的作者。")

    # 等待“下一页”按钮并检查是否存在
    try:
        next_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[@rel='next']"))
        )
        next_button.click()  # 点击“下一页”按钮
        time.sleep(2)  # 等待页面加载完成
    except:
        print("没有找到'下一页'按钮，停止爬取。")
        break  # 没有“下一页”按钮，退出循环

# 关闭WebDriver
driver.quit()
