from scrapy.spiders import Spider
from scrapy.exceptions import CloseSpider
import scrapy
import json
import re
from ..items import douban_item


data_url = "https://movie.douban.com/j/new_search_subjects?sort=U&range=0,10&tags=%E7%94%B5%E5%BD%B1&start={}&genres={}"

class douban_spider(Spider):
    name = "douban_spider"

    def __init__(self, movie_genre, start, end, max_depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = "https://movie.douban.com/j/new_search_subjects?sort=U&range=0,10&tags=%E7%94%B5%E5%BD%B1&" \
                        "start={}&genres=" + movie_genre
        self.movie_genre = movie_genre
        self.start = start
        self.next = int(start)
        self.end = int(end)
        self.start_urls = [self.base_url.format(start)]
        self.max_depth = 1
        self.info_dict = {
            '导演': 'director',
            '类型': 'genres',
            '制片国家/地区': 'region',
            'IMDb链接': 'imdb',
        }

    def parse(self, response, **kwargs):
        self.logger.info(f'now the number is {self.next}')
        self.logger.info(f'crawled url is\n{response.url}')

        json_response = json.loads(response.text)['data']
        for item in json_response:
            url = item['url']
            # url = "https://movie.douban.com/subject/26323031/"
            yield scrapy.Request(url, callback=self.parse_page, dont_filter=True)
            # return None

        self.next += 20
        if self.next >= self.end:
            self.logger.info(f'{self.movie_genre} now done! from {self.start} to {self.next} >= {self.end}')
            return None
        next_url = self.base_url.format(self.next)
        yield scrapy.Request(next_url, callback=self.parse, dont_filter=True)

    def remove_some(self, text):
        if isinstance(text, str):
            pattern = "[\s\n\r]"
            text = re.sub(pattern, '', text)
            pattern = "<[^>]+>"
            text = re.sub(pattern, '', text)
        return text

    def parse_page(self, response):
        self.logger.info(f'crawled url is\n{response.url}')
        pipeitem = douban_item()
        if 'ddepth' in response.meta:
            ddepth = response.meta['ddepth'] + 1
        else:
            pipeitem['targets'] = ''
            ddepth = 1
        pic_url = response.xpath("//script[@type='application/ld+json']/text()").getall()
        if pic_url and len(pic_url) > 0:
            pic_url = pic_url[0]
            pic_url = json.loads(pic_url, strict=False)
            if 'image' in pic_url:
                pic_url = pic_url['image']
            else:
                pic_url = 'NULL'
        else:
            pic_url = 'NULL'

        brief = response.xpath("//span[@property='v:summary']/text()").getall()
        if brief and len(brief) > 0:
            brief_text = ""
            for temp in brief:
                brief_text += temp
            brief_text = self.remove_some(brief_text)
        else:
            brief_text = 'NULL'


        name = response.xpath("//span[@property='v:itemreviewed']/text()").getall()[0]
        rate = response.xpath("//strong[@class='ll rating_num']/text()").getall()
        if rate and len(rate) == 1:
            rate = rate[0]
        else:
            rate = -1
        info = response.xpath("//div[@id='info']").getall()[0]
        info = re.sub("<[^>]+>", '', info)
        recommends = response.xpath("//div[@class='recommendations-bd']").getall()[0]

        pipeitem['brief'] = brief_text
        pipeitem['name'] = self.remove_some(name)
        pipeitem['rate'] = self.remove_some(rate)
        pipeitem['page_url'] = response.url
        pipeitem['pic_url'] = pic_url


        for c, e in self.info_dict.items():
            pattern = f"{c}.*[\n\r]"
            temp = re.findall(pattern, info)
            if temp and len(temp) == 1:
                temp = temp[0][len(c)+1:]
            else:
                temp = 'NULL'
            pipeitem[e] = self.remove_some(temp)

        if ddepth <= self.max_depth:
            pattern1 = "https://.*\?"
            recommends_page_urls = re.findall(pattern1, recommends)
            for ta in recommends_page_urls:
                ta = ta[:-1]
                yield scrapy.Request(ta, callback=self.parse_page, dont_filter=True,
                                         meta={'last': pipeitem, 'ddepth': ddepth})
        else:
            last = response.meta['last']
            last['targets'] += (pipeitem['imdb'] + ';')
            pipeitem['targets'] = 'NULL'

        yield pipeitem

