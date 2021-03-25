# from scrapy.spiders import Spider
# import scrapy
# import json
# import re
# from ..items import ScrapyspiderItem
#
#
# class wuyi_spider(Spider):
#     name = "wuyi_spider"
#
#     def __init__(self, job_type, start_url, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.base_url = start_url
#         self.page_number = 1
#         self.final_stop_number = 15
#         self.start_urls = [self.base_url.format(self.page_number)]
#         self.job_type = job_type
#
#     def parse(self, response, **kwargs):
#         self.logger.info(f'now the page number is {self.page_number}')
#         self.logger.info(f'crawled url is\n{response.url}')
#
#         items = response.xpath('//script[@type="text/javascript"]/text()')  # xpath返回选择器对象
#         for item in items:
#             jsonText = item.extract()  # extract函数 解析出选择器中的文本内容
#             if jsonText.find("__SEARCH_RESULT__") > 0:
#                 jsonData = jsonText.split("= ")[1]
#                 jobDict = json.loads(jsonData)
#                 jobDict = jobDict["engine_search_result"]
#                 for job in jobDict:
#                     secondary_page = job["job_href"]
#                     pipeitem = ScrapyspiderItem()
#                     pipeitem['job_name'] = job["job_name"].strip()
#                     pipeitem['job_company'] = job["company_name"].strip()
#                     pipeitem['job_salary'] = job["providesalary_text"].strip()
#                     pipeitem['job_date'] = job["updatedate"].strip()
#                     pipeitem['job_address'] = job["workarea_text"].strip()
#                     pipeitem['job_url'] = secondary_page.strip()
#                     pipeitem['job_type'] = self.job_type
#
#                     # the job_info is in the secondary page
#                     yield scrapy.Request(secondary_page, callback=self.parse_secondary_page,
#                                          dont_filter=True, meta={'pipeitem': pipeitem})
#         self.page_number += 1
#         if self.page_number > self.final_stop_number or len(jobDict) == 0:
#             return None
#         next_url = self.base_url.format(self.page_number)
#         yield scrapy.Request(next_url, callback=self.parse, dont_filter=True)
#
#     def parse_secondary_page(self, response):
#         self.logger.info(f'crawled url is\n{response.url}')
#         pipeitem = response.meta['pipeitem']
#         items = response.xpath("//div[@class='bmsg job_msg inbox']").getall()
#         if items:
#             for info in items:
#                 if not info.isspace():
#                     pattern = "(?:<[^>]+>)|\s"
#                     info = re.sub(pattern, "", info)
#                     pipeitem['job_info'] = info
#                     yield pipeitem  # 向管道输出对象
#                     return None
#         pipeitem['job_info'] = ''
#         yield pipeitem
#
#
# class douban_spider(Spider):
#     name = "douban_250"
#     start_urls = ["https://movie.douban.com/top250"]
#     base_url = "https://movie.douban.com/top250?start={}&filter="
#     start_number = 0
#     interval = 25
#     final_number = 10
#
#     def parse(self, response, **kwargs):
#         print(f'now page number is {self.start_number + 1}')
#         items = response.xpath("//div[@class='item']/div[@class='info']/div[@class='hd']/a/span[@class='title']/text()")
#         for item in items:
#             print(item.extract())
#
#         self.start_number += 1
#         if self.start_number > self.final_number:
#             return None
#         next_url = self.base_url.format(self.start_number * self.interval)
#         yield scrapy.Request(next_url, callback=self.parse, dont_filter=True)


# class ScrapyspiderPipeline:
#
#     def open_spider(self, spider):
#         self.db = job_db(config='./database.json', logger=spider.logger)
#         self.logger = spider.logger
#         self.number = 0
#
#     def process_item(self, item, spider):
#         self.number += 1
#
#         info = f"got {self.number}\t{item['job_name']} {item['job_salary']}\t{item['job_company']} {item['job_address']} {item['job_type']}"
#
#         self.logger.info(info)
#
#         params = [
#             item['job_name'], item['job_company'], item['job_address'], item['job_salary'], item['job_type'],
#             item['job_date'], item['job_url'], item['job_info']
#         ]
#
#         # process and store in the database
#         self.db.insert_item(params)
#
#         return item
#
#     def close_spider(self, spider):
#         self.db.close()