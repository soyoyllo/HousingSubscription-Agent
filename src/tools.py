import requests
import datetime
import time
import re
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
from smolagents import Tool
from typing import Dict
import urllib.parse
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import CharacterTextSplitter

from kiwipiepy import Kiwi
kiwi = Kiwi()

def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)] 

YOUTH_NOTICE_URL = 'https://soco.seoul.go.kr/youth/pgm/home/yohome/bbsListJson.json'
YOUTH_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://soco.seoul.go.kr/youth/bbs/BMSR00015/list.do?menuNo=400008",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Origin": "https://soco.seoul.go.kr",
}

SH_BASE_URL = "https://www.i-sh.co.kr/"
SH_FILE_URL = "https://www.i-sh.co.kr/main/com/file/innoFD.do"
LH_PREVIEW_URL = "https://apply.lh.or.kr/view/viewer/document/docviewer.do"
LH_FILE_URL = "https://apply.lh.or.kr/lhapply/lhFile.do"
YOUTH_BASE_URL = "https://soco.seoul.go.kr"
YOUTH_PREVIEW_URL = "https://seoul.viewstory.net/previewAjax.do"
        
APPLICATION_TYPE_MAP = {
    "1": "공공임대 청년안심주택",
    "2": "민간임대 청년안심주택",
}

class OpenAPIGuideRetriever(Tool):
    name = "open_api_guide_retriever"
    description = (
        """
        Find the most appropriate OpenAPI usage guide document for the prompt.
        This document includes the Call Back URL for API calls, request message specifications, and response message specifications.
        If you are asked to find the Call back URL and Request parameters for the public API to use 분양임대공고조회, search with {query : 분양임대공고조회}.
        """
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"    
    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, preprocess_func=kiwi_tokenize).from_documents(
            docs, k=1
        )
        
    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n{doc.metadata}" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )
    
class GetResponse(Tool):
    name = "get_response"
    description = (
        """
        A tool that uses the request function to receive and return the response.
        """
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "Call Back URL to invoke the function.",
        },
        "request_parameter": {
            "type": "object",
            "description": "Parameters required to call the API",
        }    
    }
    output_type = "object"    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, url: str, request_parameter: dict) -> object:
        assert isinstance(url, str), "URL must be a string"
        assert isinstance(request_parameter, object), "Request parameter must be a object"
        try:
            request_parameter['serviceKey'] = os.environ['OPEN_DATA_API_KEY']
            response = requests.get(url, params=request_parameter)
            return response.json()
        except Exception as e:
            return f"Error occurred: {str(e)}"

class SH_NoticeScraper(Tool):
    name = "sh_notice_scraper"
    description = (
        """A tool for scraping the list of posts from a website's bulletin board and returning the information of posts that match the conditions.
        Additionally, it takes conditions (starting page number, maximum number of pages to scrape) as input and scrapes the table pages accordingly.
        output = {
            "번호": str,
            "청약유형": str,
            "공고명": str,
            "공고게시일": str,
            "공고상태": str,
            "발표일": str,
            "담당부서": str,
            "링크": str,
        }
        """
    )
    inputs = {
        "url": {
            "type": "string",
            "description": 
            """
                requests를 보낼 게시판이 있는 웹 페이지 URL이며 조건에 따라 달라짐. 
                (예시2: keyword가 매입임대 일 때 https://housing.seoul.go.kr/site/main/sh/publicLease/list?sc=titl&startDate=&endDate=&splyCd=&sv=장기미임대)
            """
        },
        "keyword": {
            "type": "string",
            "description": "게시글 제목에 포함되어야 하는 공고 유형 키워드 (예: '발표','합격자'). 빈 문자열이면 청약 공고문을 반환."
        },
        "max_pages": {
            "type": "integer",
            "description": "최대로 스크래핑할 페이지 수 defalut는 5"
        }
    }
    output_type = "array"
    
    def forward(self, url: str, keyword : str, max_pages: int) -> list:
        today = datetime.datetime.now()
        results = []
        url  += f'&sv={keyword}'
        if not max_pages: max_pages = 1
        for page in range(1, max_pages + 1):
            # URL에 'cp' 파라미터를 추가하여 페이지 번호를 지정 (게시판 페이지네이션 기준)
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            qs["cp"] = [str(page)]
            new_query = urlencode(qs, doseq=True)
            page_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
            
            response = requests.get(page_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 게시글 목록은 <div class="board-list"> 내부의 <table>에 있습니다.
            board_div = soup.find("div", class_="board-list")
            if board_div:
                table = board_div.find("table")
                if table:
                    tbody = table.find("tbody")
                    if tbody:
                        rows = tbody.find_all("tr")
                        for row in rows:
                            cols = row.find_all("td")
                            if len(cols) < 7:
                                continue
                            number = cols[0].get_text(strip=True)
                            notice_type = cols[1].get_text(strip=True)
                            
                            # 공고명: 텍스트와 링크가 있을 수 있으므로 <a> 태그가 있으면 그 텍스트 사용
                            title_td = cols[2]
                            title_link = title_td.find("a")
                            title = title_link.get_text(strip=True) if title_link else title_td.get_text(strip=True)
                            
                            # keyword가 공고명에 포함되어 있는지 확인
                            # if keyword and keyword not in title:
                            #     continue
                            
                            pub_date = cols[3].get_text(strip=True)
                            announce_date = cols[4].get_text(strip=True)
                            
                            # 링크: "바로가기" 버튼이 있는 <a> 태그를 찾아 URL을 구성
                            link_td = cols[6]
                            a_tag = link_td.find("a")
                            link = urljoin(page_url, a_tag.get("href")) if a_tag and a_tag.get("href") else ""
                            pub_date_obj = datetime.datetime.strptime(pub_date, "%Y-%m-%d")

                            if notice_type == "장기전세":
                                deadline = pub_date_obj + datetime.timedelta(weeks=4)
                            else:
                                deadline = pub_date_obj + datetime.timedelta(weeks=2)
                            
                            if today < deadline:
                                status = "접수중"
                            else:
                                status = "접수마감"
                            
                            result_item = {
                                #"번호": number,
                                "청약유형": notice_type,
                                "공고명": title,
                                "공고게시일": pub_date,
                                "공고상태" : status,
                                #"발표일": announce_date,
                                "공고URL": link,
                                "공고지역": '서울특별시',
                                "공고유형" : '임대주택'
                            }
                            results.append(result_item)
            else:
                # 페이지에 게시글 목록이 없으면 반복 종료
                break
        
        return results

class YouthNoticeScraper(Tool):
    name = "youth_notice_scraper"
    description = (
        "이 도구는 서울특별시 청년안심주택 웹페이지 모집공고의 API 엔드포인트를 호출하여, 게시판 목록(모집공고글)을 스크래핑하고 "
        "각 게시물의 정보를 정리하여 반환합니다. 페이지네이션을 통해 최대 지정된 페이지 수 만큼 데이터를 수집합니다."
    )
    inputs = {
        "max_pages": {
            "type": "integer",
            "description": "스크래핑할 최대 페이지 수 (default: 5)"
        },
        "search_params": {
            "type": "object",
            "description": "검색 조건을 포함한 객체 search_params={'지역' : str, '기간' : str, '청약유형' : str, '공고상태' : str, 'keyword' : str}"
        }
    }
    output_type = "array"

    def fetch_announcements(self, page_index: int) -> dict:
        payload = {
            "bbsId": "BMSR00015",
            "pageIndex": page_index,
            "searchCondition": "",
            "searchAdresGu": "",
            "searchKeyword": ""
        }
        response = requests.post(YOUTH_NOTICE_URL, headers=YOUTH_HEADERS, data=urlencode(payload))
        response.raise_for_status()
        return response.json()

    def crawl_all_announcements(self, max_pages: int) -> list:
        today = datetime.datetime.now()
        all_items = []
        first_response = self.fetch_announcements(1)
        total_pages = first_response.get("pagingInfo", {}).get("totPage", max_pages)
        pages_to_scrape = min(total_pages, max_pages)

        for page in range(1, pages_to_scrape + 1):
            data = self.fetch_announcements(page)
            items = data.get("resultList", [])
            for item in items:
                # 인라인으로 게시글 정보를 가공
                application_type_code = item.get("optn2")
                application_type = APPLICATION_TYPE_MAP.get(application_type_code, "기타")
                status = "접수중" if today.date() < datetime.datetime.strptime(item.get("optn4"), "%Y-%m-%d").date() else "접수마감"
                parsed_item = {
                    "공고명": item.get("nttSj"),
                    "공고게시일": item.get("optn1"),
                    "모집마감일": item.get("optn4"),
                    "청약유형": application_type,
                 #   "시행사": item.get("optn3"),
                 #   "게시글 ID": item.get("boardId"),
                    "공고상태": status,
                    "공고URL": f"https://soco.seoul.go.kr/youth/bbs/{item.get('bbsId')}/view.do?boardId={item.get('boardId')}&menuNo=400008",
                    "공고지역" : "서울특별시",
                    "공고유형" : '임대주택'
                }
                all_items.append(parsed_item)
            time.sleep(0.5)  # 너무 빠른 요청 방지                                                                                                                                                                                                                  
        return all_items
    
    def notice_filter(self, all_items, search_params: dict) -> list:
        지역 = search_params.get('지역', '')
        기간_범위 = search_params.get('기간', '')
        공고상태_filter = search_params.get('공고상태', '')
        keyword = search_params.get('keyword', '')
        
        # 기간 파싱
        try:
            if 기간_범위 and '~' in 기간_범위:
                기간_start, 기간_end = 기간_범위.split('~')
                기간_start = datetime.datetime.strptime(기간_start.strip(), "%Y-%m-%d")
                기간_end = datetime.datetime.strptime(기간_end.strip(), "%Y-%m-%d")
            else:
                기간_start, 기간_end = datetime.datetime.min, datetime.datetime.max
        except ValueError as e:
            print(f"기간 파싱 오류: {기간_범위}, 에러: {e}")
            return []
        
        def safe_date_check(notice_date_str):
            try:
                if not notice_date_str:
                    return False
                notice_date = datetime.datetime.strptime(notice_date_str, "%Y-%m-%d")
                return 기간_start <= notice_date <= 기간_end
            except (ValueError, TypeError):
                return False
        
        filtered_notices = [
            notice for notice in all_items
            if (not 지역 or 지역 in str(notice.get('공고지역', '')))
            and safe_date_check(notice.get('공고게시일'))
            and (not 공고상태_filter or notice.get('공고상태') == 공고상태_filter)
            and (not keyword or keyword in str(notice.get('공고명', '')))
        ]
        
        return filtered_notices

    def forward(self, max_pages: int, search_params : dict) -> list:
        all_items = self.crawl_all_announcements(max_pages)
        return self.notice_filter(all_items, search_params)
    
def get_page_source(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print("Requests 실패, Selenium으로 전환합니다. 에러:", e)
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        html = driver.page_source
        driver.quit()
        return html
    
class NoticeFileScraper(Tool):
    name = "notice_file_scraper"
    description = (
        "이 도구는 웹 페이지를 방문하여 공고 일정과 공고문의 PDF 파일 다운로드 링크와 바로보기 URL를 추출합니다."
        "청년안심주택(Youth)의 공고문 상세 URL일 경우 웹 페이지 내에 타이틀이 '바로보기'인 앵커 태그를 찾은 후 extract_links_from_youth(anchors) 함수를 사용해서 다운로드 링크와 PDF 파일 바로보기 URL을 반환합니다."
        "SH의 공고문 상세 URL일 경우 웹 페이지 내에 클래스 명이 'gs0401tr'인 tr 태그들을 찾은 후 extract_links_from_sh(trs) 함수를 사용해서 다운로드 링크와 PDF 파일 미리보기 URL을 반환합니다."
        "LH의 공고문 상세 URL일 경우 웹 페이지 내에 클래스 명이 'bbsV_link file'인 ul 태그들을 찾은 후 extract_links_from_lh(uls) 함수를 사용해서 다운로드 링크와 PDF 파일 바로보기 URL을 반환합니다."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "PDF 파일 링크를 포함한 웹 페이지의 URL",
        }
    }
    output_type = "object"
    
    def __init__(self):
        self.base_sh_url = "https://www.i-sh.co.kr/"
        self.sh_file_url = "https://www.i-sh.co.kr/main/com/file/innoFD.do"
        self.lh_preview_url = "https://apply.lh.or.kr/view/viewer/document/docviewer.do"
        self.lh_file_url = "https://apply.lh.or.kr/lhapply/lhFile.do"
        self.base_youth_url = "https://soco.seoul.go.kr"
        self.youth_preview_url = "https://seoul.viewstory.net/previewAjax.do"

    def construct_url(self, base_url: str, params: dict) -> str:
        """
        base_url과 파라미터를 받아, url 주소를 구성하여 반환합니다.

        Args:
            base_url (str): 페이지의 기본 URL.
            params (dict): 쿼리 파라미터를 포함한 딕셔너리.

        Returns:
            str: 인코딩된 쿼리 파라미터를 포함한 URL.
        """
        # urllib.parse.urlencode를 사용하여 모든 파라미터를 URL 인코딩
        query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        return f"{base_url}?{query_string}"
    
    def extract_links_from_youth(self, anchors):
        pattern = r"previewAjax\('([^']+)','([^']+)'\)"
        download_link = None
        preview_url = None

        for anchor in anchors:
            if '공고' in anchor['onclick']:
                match = re.search(pattern, anchor['onclick'])
                if match:
                    download_link = self.base_youth_url + match.group(1)
                    filename = match.group(2)
        if not filename:
            print("다운로드 및 미리보기 링크를 찾을 수 없습니다.")
            return {"download_link": None, "preview_url": None}

        params = {
            "apikey": "RBKLXXZDMUBCHNHOEKZTMG",
            "cc": "sg_094",
            "url": download_link,
            "fileName": filename
        }
        preview_url = self.construct_url(self.youth_preview_url, params)

        return {"download_link": download_link, "preview_url": preview_url}
    
    def extract_links_from_sh(self, trs):
        for tr in trs:
            td = next((td for td in tr.find_all("td") if "공고" in td.get_text()), None)
            if td:
                anchor = tr.find("a", title="파일 미리보기")
                if anchor:
                    preview_url = self.base_sh_url + anchor.get("href")
                    parsed = urllib.parse.urlparse(preview_url)
                    query_params = urllib.parse.parse_qs(parsed.query)

                    params = {
                        "brdId": query_params.get("brd_id", [""])[0],
                        "seq": query_params.get("seq", [""])[0],
                        "fileTp": query_params.get("data_tp", [""])[0],
                        "fileSeq": query_params.get("file_seq", [""])[0]
                    }

                    download_link = self.construct_url(self.sh_file_url, params)
                    return {"download_link": download_link, "preview_url": preview_url}
        print("공고문 다운로드 및 미리보기 링크를 찾을 수 없습니다.")
        return {"download_link": None, "preview_url": None}

    def extract_links_from_lh(self, uls):
        preview_url = self.lh_preview_url
        for ul in uls:
            for li in ul.find_all('li'):
                text = li.get_text()
                if '.pdf' in text and '공고' in text:
                    anchor = li.select('a')
                    file_id = ''.join(filter(str.isdigit, anchor[0]['href']))
                    matches = re.findall(r"docViewer\('([^']+)', '([^']+)', '([^']+)'\)", anchor[1]['onclick'])[0]
                    params = dict(zip(["filepath", "filename", "fileext"], matches))
                    preview_url = self.construct_url(preview_url, params)
                    download_link = f"{self.lh_file_url}?fileid={file_id}"
                    return {"download_link": download_link, "preview_url": preview_url}
        
        print("공고문 다운로드 및 미리보기 링크를 찾을 수 없습니다.")
        return {"download_link": None, "preview_url": None}
    
    def extract_supply_schedule(self, soup) -> str:
        schedule_container = soup.find(lambda tag: tag.name in ["div", "section", "table"] and "공급일정" in tag.get_text())
        if not schedule_container:
            return ""
        container_text = schedule_container.get_text()
        start_index = container_text.find("공급일정")
        container_text = container_text[start_index-1:] if start_index != -1 else container_text

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=220, chunk_overlap=0)
        chunks = text_splitter.split_text(container_text)
        supply_schedule = "\n".join(chunk for chunk in chunks if "공급일정" in chunk)

        return supply_schedule

    def forward(self, url: str) -> Dict[str, str]:
        html = get_page_source(url)
        soup = BeautifulSoup(html, "html.parser")
        links = {}
        
        # 페이지 내 모든 앵커 태그 검색
        anchors = soup.select('a[title="바로보기"]')
        trs = soup.find_all("tr", class_="gs0401tr")
        uls = soup.select("ul", class_="bbsV_link file")
        
        if anchors:  # 바로보기 태그가 있을 때만 실행
            links[url] = self.extract_links_from_youth(anchors)
        elif trs:
            links[url] = self.extract_links_from_sh(trs)
        elif uls:  # 바로보기 태그가 있을 때만 실행
            links[url] = self.extract_links_from_lh(uls)
        else:
            print("조건에 맞는 링크를 찾지 못했습니다.")            

        suply_schedule = self.extract_supply_schedule(soup)
        links[url]['suply_schedule'] = suply_schedule
        return links


__all__ = [
    "NoticeFileScraper",
    "YouthNoticeScraper",
    "SH_NoticeScraper",
    "GetResponse",
    "OpenAPIGuideRetriever"
]
