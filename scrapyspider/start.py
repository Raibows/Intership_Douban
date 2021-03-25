from scrapy.cmdline import execute
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
parser.add_argument('--genre', type=str)
parser.add_argument('--max_depth', type=int)
args = parser.parse_args()

if __name__ == '__main__':
    movie_genre = {'剧情', '喜剧', '动作', '爱情', '科幻', '动画', '悬疑', '惊悚', '恐怖',
                   '犯罪', '同性', '音乐', '歌舞', '传记', '历史', '战争', '西部', '奇幻', '冒险', '灾难', '武侠', '情色'}
    assert args.genre in movie_genre
    assert args.start < args.end

    execute(['scrapy', 'crawl', 'douban_spider',
             '-a', f'movie_genre={args.genre}',
             '-a', f'start={args.start}',
             '-a', f'end={args.end}',
             '-a', f'max_depth={args.max_depth}',
             ])
