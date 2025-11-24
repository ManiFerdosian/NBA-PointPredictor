"""
Generate a balanced NBA dataset with realistic player distributions.
"""
import random
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def generate_realistic_game_stats(player_type, base_minutes):
    """
    Generate realistic game stats based on player type.
    
    Args:
        player_type: 'star', 'role', 'bench'
        base_minutes: Base minutes for the player
    
    Returns:
        Dictionary with game stats
    """
    if player_type == 'star':
        # Star players: 20-35 points, high usage
        points = random.randint(18, 38)
        fga = random.randint(15, 25)
        fgm = int(fga * random.uniform(0.42, 0.55))
        three_pa = random.randint(5, 12)
        three_pm = int(three_pa * random.uniform(0.30, 0.45))
        fta = random.randint(4, 12)
        ftm = int(fta * random.uniform(0.75, 0.90))
        rebounds = random.uniform(5, 12)
        assists = random.uniform(4, 10)
        
    elif player_type == 'role':
        # Role players: 8-18 points, moderate usage
        points = random.randint(6, 20)
        fga = random.randint(8, 15)
        fgm = int(fga * random.uniform(0.40, 0.50))
        three_pa = random.randint(2, 7)
        three_pm = int(three_pa * random.uniform(0.30, 0.40))
        fta = random.randint(1, 6)
        ftm = int(fta * random.uniform(0.70, 0.85))
        rebounds = random.uniform(3, 8)
        assists = random.uniform(2, 6)
        
    else:  # bench
        # Bench players: 2-12 points, low usage
        points = random.randint(1, 14)
        fga = random.randint(3, 10)
        fgm = int(fga * random.uniform(0.35, 0.48))
        three_pa = random.randint(0, 5)
        three_pm = int(three_pa * random.uniform(0.25, 0.40))
        fta = random.randint(0, 4)
        ftm = int(fta * random.uniform(0.65, 0.80))
        rebounds = random.uniform(1, 6)
        assists = random.uniform(0, 4)
    
    # Ensure points match made shots
    points_from_fg = fgm * 2 + three_pm * 3
    points_from_ft = ftm
    actual_points = points_from_fg + points_from_ft
    
    # Adjust if needed
    if abs(actual_points - points) > 2:
        points = actual_points
    
    return {
        'points': points,
        'field_goals_attempted': fga,
        'field_goals_made': fgm,
        'three_pa': three_pa,
        'three_pm': three_pm,
        'free_throws_attempted': fta,
        'free_throws_made': ftm,
        'rebounds': round(rebounds, 1),
        'assists': round(assists, 1),
        'minutes': round(base_minutes + random.uniform(-3, 3), 1)
    }


def generate_balanced_dataset():
    """Generate a balanced NBA dataset."""
    teams = ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Bucks', 'Nuggets', 
             'Suns', '76ers', 'Mavericks', 'Spurs', 'Nets', 'Raptors', 
             'Bulls', 'Cavaliers', 'Clippers']
    
    # Star players (keep existing ones, add a few more)
    star_players = [
        'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
        'Luka Doncic', 'Jayson Tatum', 'Joel Embiid', 'Devin Booker',
        'Anthony Davis', 'Jaylen Brown', 'Kyrie Irving', 'Damian Lillard',
        'Jimmy Butler', 'Trae Young', 'Donovan Mitchell'
    ]
    
    # Role players (8-18 PPG average)
    role_players = [
        'Kawhi Leonard', 'Paul George', 'Zach LaVine', 'Shai Gilgeous-Alexander',
        'Jalen Brunson', 'Karl-Anthony Towns', 'Bam Adebayo', 'LaMelo Ball',
        'Darius Garland', 'DeMar DeRozan', 'Pascal Siakam', 'Brandon Ingram',
        'CJ McCollum', 'Fred VanVleet', 'Julius Randle', 'Andrew Wiggins',
        'Klay Thompson', 'Draymond Green', 'Michael Porter Jr.', 'Jamal Murray',
        'Chet Holmgren', 'Victor Wembanyama', 'Rudy Gobert', 'Nikola Jokic',
        'Aaron Gordon', 'Tyrese Maxey', 'Tyrese Haliburton', 'Myles Turner',
        'Jaren Jackson Jr.', 'Desmond Bane', 'OG Anunoby', 'Scottie Barnes',
        'Franz Wagner', 'Paolo Banchero', 'Markelle Fultz', 'Cole Anthony',
        'Buddy Hield', 'Mikal Bridges', 'Cam Johnson', 'Nic Claxton',
        'Brook Lopez', 'Khris Middleton', 'Jrue Holiday', 'Marcus Smart',
        'Al Horford', 'Robert Williams III', 'Jerami Grant', 'Anfernee Simons',
        'Deandre Ayton', 'Jusuf Nurkic', 'Terry Rozier', 'Gordon Hayward',
        'Miles Bridges', 'Derrick White', 'Malcolm Brogdon', 'Tyler Herro',
        'Bobby Portis', 'Jordan Poole', 'Kyle Kuzma', 'Kristaps Porzingis',
        'Bradley Beal', 'Spencer Dinwiddie', 'Alperen Sengun', 'Jalen Green',
        'Kevin Porter Jr.', 'Keldon Johnson', 'Devin Vassell', 'Jonas Valanciunas',
        'Herbert Jones', 'Trey Murphy III', 'Zion Williamson', 'Jaden Ivey',
        'Cade Cunningham', 'Isaiah Stewart', 'Bojan Bogdanovic', 'Jarrett Allen',
        'Evan Mobley', 'Caris LeVert', 'Matisse Thybulle', 'Josh Giddey',
        'Lou Dort'
    ]
    
    # Bench players (2-12 PPG average)
    bench_players = [
        'Pat Connaughton', 'Grayson Allen', 'Jevon Carter', 'Thanasis Antetokounmpo',
        'MarJon Beauchamp', 'AJ Green', 'Lindell Wigginton', 'Chris Livingston',
        'Andre Jackson', 'Omari Moore', 'Tyler Cook', 'Joe Ingles',
        'Cameron Payne', 'Keita Bates-Diop', 'Drew Eubanks', 'Yuta Watanabe',
        'Chimezie Metu', 'Ish Wainright', 'Saben Lee', 'Udoka Azubuike',
        'Theo Maledon', 'Jock Landale', 'Darius Bazley', 'Isaiah Todd',
        'Kendall Brown', 'Jeremy Sochan', 'Blake Wesley', 'Dominick Barlow',
        'Charles Bassey', 'Devonte Graham', 'Garrett Temple', 'Wesley Matthews',
        'Maxi Kleber', 'Dwight Powell', 'Markieff Morris', 'Frank Ntilikina',
        'Theo Pinson', 'A.J. Lawson', 'Greg Brown III', 'Dereck Lively II',
        'Richaun Holmes', 'JaVale McGee', 'Boban Marjanovic', 'P.J. Washington',
        'Nick Richards', 'James Bouknight', 'JT Thor', 'Kai Jones',
        'Bryce McGowens', 'Moussa Diabate', 'Leaky Black', 'Amari Bailey'
    ]
    
    all_players = []
    
    # Add star players with their games
    for player in star_players:
        player_type = 'star'
        base_minutes = random.uniform(32, 38)
        num_games = random.randint(8, 12)
        all_players.append((player, player_type, base_minutes, num_games))
    
    # Add role players
    for player in role_players:
        player_type = 'role'
        base_minutes = random.uniform(22, 32)
        num_games = random.randint(5, 10)
        all_players.append((player, player_type, base_minutes, num_games))
    
    # Add bench players
    for player in bench_players:
        player_type = 'bench'
        base_minutes = random.uniform(8, 20)
        num_games = random.randint(3, 8)
        all_players.append((player, player_type, base_minutes, num_games))
    
    # Generate games
    games = []
    start_date = datetime(2024, 1, 1)
    
    for player, player_type, base_minutes, num_games in all_players:
        for i in range(num_games):
            game_date = start_date + timedelta(days=random.randint(0, 90))
            team = random.choice(teams)
            opponent = random.choice([t for t in teams if t != team])
            
            stats = generate_realistic_game_stats(player_type, base_minutes)
            
            game = {
                'game_date': game_date.strftime('%Y-%m-%d'),
                'player_name': player,
                'team': team,
                'opponent': opponent,
                'minutes': stats['minutes'],
                'points': stats['points'],
                'rebounds': stats['rebounds'],
                'assists': stats['assists'],
                'field_goals_attempted': stats['field_goals_attempted'],
                'field_goals_made': stats['field_goals_made'],
                'three_pa': stats['three_pa'],
                'three_pm': stats['three_pm'],
                'free_throws_attempted': stats['free_throws_attempted'],
                'free_throws_made': stats['free_throws_made']
            }
            games.append(game)
    
    # Create DataFrame
    df = pd.DataFrame(games)
    
    # Sort by date
    df = df.sort_values('game_date').reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    print("Generating balanced NBA dataset...")
    df = generate_balanced_dataset()
    
    # Calculate statistics
    TARGET_LINE = 30
    over_30_count = (df['points'] > TARGET_LINE).sum()
    under_30_count = (df['points'] <= TARGET_LINE).sum()
    total_games = len(df)
    
    print(f"\nDataset Statistics (Target Line: {TARGET_LINE} points):")
    print(f"Total games: {total_games}")
    print(f"Games over {TARGET_LINE} points: {over_30_count} ({over_30_count/total_games*100:.1f}%)")
    print(f"Games under {TARGET_LINE} points: {under_30_count} ({under_30_count/total_games*100:.1f}%)")
    print(f"Unique players: {df['player_name'].nunique()}")
    
    # Calculate average points per player type
    star_players = ['LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
                    'Luka Doncic', 'Jayson Tatum', 'Joel Embiid', 'Devin Booker',
                    'Anthony Davis', 'Jaylen Brown', 'Kyrie Irving', 'Damian Lillard',
                    'Jimmy Butler', 'Trae Young', 'Donovan Mitchell']
    
    star_avg = df[df['player_name'].isin(star_players)]['points'].mean()
    non_star_avg = df[~df['player_name'].isin(star_players)]['points'].mean()
    
    print(f"\nAverage points per game:")
    print(f"Star players: {star_avg:.1f}")
    print(f"Non-star players: {non_star_avg:.1f}")
    
    # Save to CSV
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "assets" / "nbaplayers_500games.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")

